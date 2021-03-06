import numpy as np

from typing import Tuple

from smg.imagesources import RGBDImageSource

from .openni_camera import OpenNICamera


class OpenNIRGBDImageSource(RGBDImageSource):
    """An RGB-D image source that wraps an OpenNI camera."""

    # CONSTRUCTOR

    def __init__(self, camera: OpenNICamera):
        """
        Construct an RGB-D image source that wraps an OpenNI camera.

        :param camera:  The OpenNI camera.
        """
        self.__camera = camera  # type: OpenNICamera

    # PUBLIC METHODS

    def get_colour_intrinsics(self) -> Tuple[float, float, float, float]:
        """
        Get the colour camera intrinsics.

        :return:    The colour camera intrinsics, as an (fx, fy, cx, cy) tuple.
        """
        return self.__camera.get_colour_intrinsics()

    def get_colour_size(self) -> Tuple[int, int]:
        """
        Get the size of the colour images.

        :return:    The size of the colour images, as a (width, height) tuple.
        """
        return self.__camera.get_colour_size()

    def get_images(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get colour and depth images from the image source.

        :return:    A tuple consisting of a colour image and a depth image from the image source (in that order).
        """
        return self.__camera.get_images()

    def terminate(self) -> None:
        """Tell the image source to terminate."""
        self.__camera.terminate()
