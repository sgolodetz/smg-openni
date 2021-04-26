import math
import numpy as np

from func_timeout import func_timeout, FunctionTimedOut
from openni import openni2
from typing import Optional, Tuple


class OpenNICamera:
    """An OpenNI camera wrapper."""

    # CONSTRUCTOR

    def __init__(self, *, debug: bool = False, manage_openni: bool = True, mirror_images: bool = False):
        """
        Construct an OpenNI camera.

        :param debug:           Whether to print out debug messages.
        :param mirror_images:   Whether to horizontally mirror the depth and colour images (needed for a Kinect).
        """
        self.__manage_openni = manage_openni  # type: bool
        self.__mirror_images = mirror_images  # type: bool
        self.__terminated = False  # type: bool

        # If we're managing OpenNI, initialise it.
        if manage_openni:
            openni2.initialize()

        # Open the device (i.e. the camera).
        self.__device = openni2.Device.open_any()  # type: openni2.Device
        if debug:
            print("OpenNI Device Info: {}".format(self.__device.get_device_info()))

        # Create and start the depth stream, and tell OpenNI to register the depth and colour images.
        # FIXME: We should handle the error case of create_depth_stream returning None.
        self.__depth_stream = self.__device.create_depth_stream()  # type: Optional[openni2.VideoStream]
        self.__depth_stream.start()
        self.__device.set_image_registration_mode(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR)
        if debug:
            print("Depth Video Mode: {}".format(self.__depth_stream.get_video_mode()))

        # Create and start the colour stream.
        # FIXME: We should handle the error case of create_color_stream returning None.
        self.__colour_stream = self.__device.create_color_stream()  # type: Optional[openni2.VideoStream]
        self.__colour_stream.start()
        if debug:
            print("Colour Video Mode: {}".format(self.__colour_stream.get_video_mode()))

        # Determine the depth camera intrinsics.
        # FIXME: The height and width shouldn't be hard-coded like this.
        self.__depth_height = 480  # type: int
        self.__depth_width = 640  # type: int

        self.__depth_intrinsics = OpenNICamera.__make_intrinsics(
            self.__depth_width, self.__depth_height,
            self.__depth_stream.get_horizontal_fov(),
            self.__depth_stream.get_vertical_fov()
        )  # type: Tuple[float, float, float, float]

        # Determine the colour camera intrinsics.
        # FIXME: The height and width shouldn't be hard-coded like this.
        self.__colour_height = 480  # type: int
        self.__colour_width = 640  # type: int

        self.__colour_intrinsics = OpenNICamera.__make_intrinsics(
            self.__colour_width, self.__colour_height,
            self.__colour_stream.get_horizontal_fov(),
            self.__colour_stream.get_vertical_fov()
        )  # type: Tuple[float, float, float, float]

    # DESTRUCTOR

    def __del__(self):
        """Destroy the OpenNI camera."""
        self.terminate()

    # SPECIAL METHODS

    def __enter__(self):
        """No-op (needed to allow the camera's lifetime to be managed by a with statement)."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Destroy the camera at the end of the with statement that's used to manage its lifetime."""
        self.terminate()

    # PUBLIC METHODS

    def get_colour_intrinsics(self) -> Tuple[float, float, float, float]:
        """
        Get the colour camera intrinsics.

        :return:    The colour camera intrinsics, as an (fx, fy, cx, cy) tuple.
        """
        return self.__colour_intrinsics

    def get_colour_size(self) -> Tuple[int, int]:
        """
        Get the size of the colour images.

        :return:    The size of the colour images, as a (width, height) tuple.
        """
        return self.__colour_width, self.__colour_height

    def get_depth_intrinsics(self) -> Tuple[float, float, float, float]:
        """
        Get the depth camera intrinsics.

        :return:    The depth camera intrinsics, as an (fx, fy, cx, cy) tuple.
        """
        return self.__depth_intrinsics

    def get_depth_size(self) -> Tuple[int, int]:
        """
        Get the size of the depth images.

        :return:    The size of the depth images, as a (width, height) tuple.
        """
        return self.__depth_width, self.__depth_height

    def get_images(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get colour and depth images from the camera.

        :return:    A tuple consisting of a colour image and a depth image from the camera (in that order).
        """
        depth_frame = self.__depth_stream.read_frame()
        colour_frame = self.__colour_stream.read_frame()

        depth_data = depth_frame.get_buffer_as_uint16()
        colour_data = colour_frame.get_buffer_as_uint8()

        depth_image = np.frombuffer(depth_data, dtype=np.uint16) / 1000  # type: np.ndarray
        depth_image = depth_image.reshape((self.__depth_height, self.__depth_width))

        colour_image = np.frombuffer(colour_data, dtype=np.uint8)  # type: np.ndarray
        colour_image = colour_image.reshape((self.__colour_height, self.__colour_width, 3))
        colour_image = OpenNICamera.__flip_channels(colour_image)

        # Mirror both the depth and colour images horizontally if necessary.
        if self.__mirror_images:
            depth_image = np.ascontiguousarray(np.fliplr(depth_image))
            colour_image = np.ascontiguousarray(np.fliplr(colour_image))

        return colour_image, depth_image

    def terminate(self) -> None:
        """Destroy the OpenNI camera."""
        if not self.__terminated:
            # Stop the depth and colour streams.
            self.__depth_stream.stop()
            self.__colour_stream.stop()

            # If we're managing OpenNI, try to unload it. If it won't unload after 2s, don't bother.
            if self.__manage_openni:
                try:
                    func_timeout(2, openni2.unload)
                except FunctionTimedOut:
                    pass

            self.__terminated = True

    # PRIVATE STATIC METHODS

    @staticmethod
    def __flip_channels(image: np.ndarray) -> np.ndarray:
        """
        Convert a BGR image to RGB, or vice-versa.

        :param image:   The input image.
        :return:        The output image.
        """
        return np.ascontiguousarray(image[:, :, [2, 1, 0]])

    @staticmethod
    def __make_intrinsics(width: int, height: int, hfov: float, vfov: float) -> Tuple[float, float, float, float]:
        """
        Make a set of camera intrinsics.

        :param width:   The width of the images produced by the camera (in pixels).
        :param height:  The height of the images produced by the camera (in pixels).
        :param hfov:    The horizontal field of view of the camera.
        :param vfov:    The vertical field of view of the camera.
        :return:        The camera intrinsics, in the form (fx, fy, cx, cy).
        """
        # Note: This is based on the code in OpenNIEngine.cpp in InfiniTAM.
        fx = width / (2 * math.tan(hfov / 2))
        fy = height / (2 * math.tan(vfov / 2))
        cx = width / 2
        cy = height / 2
        return fx, fy, cx, cy
