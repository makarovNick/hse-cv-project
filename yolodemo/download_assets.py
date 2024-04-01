import os
import shutil
import tarfile
from pathlib import Path

import gdown
import requests

# Get the directory of the current script to construct relative paths
script_dir = Path(__file__).parent


def download_and_extract(url, relative_output_path):
    # Construct the full output path relative to the script directory
    output_path = script_dir / relative_output_path

    # Download the file
    gdown.download(url, str(output_path), quiet=False)

    # Extract the file
    with tarfile.open(output_path) as tar:
        tar.extractall(path=script_dir / "assets")
    # Remove the tar.gz file after extraction
    output_path.unlink()


def download_file(url, relative_output_path):
    # Construct the full output path relative to the script directory
    output_path = script_dir / "assets" / relative_output_path

    response = requests.get(url)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as file:
        file.write(response.content)


def download_assets():
    print("Downloading files...")
    download_and_extract(
        "https://drive.google.com/uc?id=13z-6INu_203A97p_pMvsQux0qnu-WV8A",
        "resources.tar.gz",
    )
    download_and_extract(
        "https://drive.google.com/uc?id=1IQIaKwQ61RQCDAQuNmLHh0fqzRw8wEVT",
        "checkpoint.tar.gz",
    )
    download_and_extract(
        "https://drive.google.com/uc?id=11pbvvFJLAS7qLv68odP80Ea6CYkjlQGi",
        "saved_model.tar.gz",
    )

    print("Downloading voc.names...")
    download_file(
        "https://raw.githubusercontent.com/pjreddie/darknet/master/data/voc.names",
        Path("data/classes/voc.names"),
    )

    # Ensure the `convert_graph_def_to_saved_model` function also uses paths relative to `script_dir`
    print("Converting TF graph...")
    saved_model_dir = script_dir / "assets" / "saved_model"
    shutil.rmtree(saved_model_dir, ignore_errors=True)

    from yolodemo.freeze_graph_to_saved_model import convert_graph_def_to_saved_model

    convert_graph_def_to_saved_model(
        str(saved_model_dir),
        str(script_dir / "assets" / "yolov3_nano_416.pb"),
        "input/input_data",
        ["pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"],
    )

    print("Downloading sample video...")
    download_file(
        "https://github.com/YunYang1994/tensorflow-yolov3/raw/master/docs/images/road.mp4",
        str(script_dir / "assets" / "road.mp4"),
    )
    download_file(
        "https://djl.ai/examples/src/test/resources/dog_bike_car.jpg",
        str(script_dir / "assets" / "test.jpg"),
    )
    print("Files ready.")


if __name__ == "__main__":
    download_assets()
