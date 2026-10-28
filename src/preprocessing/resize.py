import cv2


def resize_image(image, width, height):
    """Resize an image to the given width and height."""
    return cv2.resize(image, (width, height))