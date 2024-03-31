# HSE Computer Vision Project

This project implements an object detection system using the YOLOv3 model. It processes video files to detect objects and displays the results in real-time.

## Installation

You can install this package directly from the repository or by cloning and building it locally. Below are the instructions for both methods.

### Direct Installation from GitHub

To install the package directly from GitHub, run:

```bash
pip install git+https://github.com/makarovNick/hse-cv-project.git
```

### Cloning and Installing Locally

Alternatively, you can clone the repository and either use `poetry` or build the package manually with Python:

1. Clone the repository:

    ```bash
    git clone https://github.com/makarovNick/hse-cv-project.git
    cd hse-cv-project
    ```

2. Install using Poetry:

   If you have Poetry installed, you can set up the project and its dependencies by running:

    ```bash
    poetry install
    ```

   This will create a virtual environment and install all required dependencies.

3. Build and Install with Python:

   To build the project manually and install it, run:

    ```bash
    python -m build
    pip install dist/*.whl
    ```

   This creates a wheel distribution in the `dist/` directory, which can then be installed with `pip`.

## Usage

After installation, you can run the application using the `yolov3` command followed by optional arguments for specifying the input and output video files.

```bash
yolov3 --input path/to/input/video.mp4 --output path/to/output/video.mp4
```

If no arguments are provided, the application will default to processing `road.mp4` and saving the result to `out.mp4`.
