import cv2
import numpy as np
import pytest
from yolodemo.demo import run_detection_single_image


# Function to generate a random image
def generate_random_image(width, height, output_path):
    """
    Generates a random RGB image of specified dimensions and saves it to the output path.

    Args:
    - width (int): Width of the generated image.
    - height (int): Height of the generated image.
    - output_path (str): Path to save the generated image.
    """
    # Generate a random array of shape (height, width, 3) with values between 0 and 255
    random_image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    # Save the image
    cv2.imwrite(output_path, random_image)


@pytest.fixture(scope="session")
def random_image_path(tmp_path_factory):
    """
    Pytest fixture to generate a random image and provide its file path for testing.

    Returns:
    - str: The file path of the generated random image.
    """
    temp_dir = tmp_path_factory.mktemp("data")
    image_path = temp_dir / "random_image.jpg"
    # Generate a random image of dimensions 640x480
    generate_random_image(640, 480, str(image_path))
    return str(image_path)


def test_detection_no_error(random_image_path):
    """
    Test that object detection runs without errors on a randomly generated image.

    Args:
    - random_image_path (str): Fixture providing the file path of the generated random image.
    """
    try:
        bboxes = run_detection_single_image(random_image_path)
        # Optionally, assert something about the output, such as it being a list.
        assert isinstance(bboxes, list), "Expected a list of bounding boxes"
    except Exception as e:
        pytest.fail(f"Object detection failed with an error: {e}")
