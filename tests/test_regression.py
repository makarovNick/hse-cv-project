import cv2
import numpy as np
import pytest
from yolodemo.demo import run_detection_single_image

# Define EXPECTED_BBOXES based on the known position and size of the rectangle
EXPECTED_BBOXES = [
    np.array([268.43557739, 149.22737122, 382.36212158, 275.9475708, 0.40147233, 19.0])
]


def generate_specific_image(
    image_path, image_size=(640, 480), rectangle_size=(100, 200)
):
    """
    Generates an image with a single central black rectangle on a white background.

    Args:
    - image_path (str): Path where the image will be saved.
    - image_size (tuple): The dimensions of the image (width, height).
    - rectangle_size (tuple): The dimensions of the rectangle (width, height).
    """
    # Create a white background
    image = np.full((image_size[1], image_size[0], 3), 255, dtype=np.uint8)

    # Calculate the top-left corner of the rectangle
    x_start = (image_size[0] - rectangle_size[0]) // 2
    y_start = (image_size[1] - rectangle_size[1]) // 2

    # Draw the rectangle
    cv2.rectangle(
        image,
        (x_start, y_start),
        (x_start + rectangle_size[0], y_start + rectangle_size[1]),
        (0, 0, 0),
        -1,
    )

    # Save the image
    cv2.imwrite(image_path, image)


@pytest.fixture
def specific_image_path(tmp_path_factory):
    """
    Creates a specific test image and provides its file path.
    """
    temp_dir = tmp_path_factory.mktemp("data")
    image_path = temp_dir / "specific_image.jpg"
    generate_specific_image(str(image_path))
    return str(image_path)


def test_single_object_regression(specific_image_path):
    """
    Regression test for a specific generated image with a single object.
    """
    detected_bboxes = run_detection_single_image(specific_image_path)

    # Here you can implement more sophisticated comparison between detected_bboxes and EXPECTED_BBOXES
    assert len(detected_bboxes) == len(
        EXPECTED_BBOXES
    ), "The number of detected bounding boxes does not match the expected number."
    # Example simple comparison, assuming the ordering and exact match of bounding boxes
    print(detected_bboxes)
    for detected, expected in zip(detected_bboxes, EXPECTED_BBOXES):
        assert detected == pytest.approx(
            expected, abs=1e-3
        ), "Detected bounding box does not match the expected value."
