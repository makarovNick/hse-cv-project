import tarfile
import gdown
import requests
from pathlib import Path
from yolodemo.freeze_graph_to_saved_model import convert_graph_def_to_saved_model
import shutil

def download_and_extract(url, output_path):
    # Download the file
    gdown.download(url, output_path, quiet=False)

    # Extract the file
    with tarfile.open(output_path) as tar:
        tar.extractall(path=".")
    # Remove the tar.gz file after extraction
    Path(output_path).unlink()


def download_file(url, output_path):
    response = requests.get(url)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as file:
        file.write(response.content)


def download_assets():
    print("Downloading files...")
    download_and_extract("https://drive.google.com/uc?id=13z-6INu_203A97p_pMvsQux0qnu-WV8A", "resources.tar.gz")
    download_and_extract("https://drive.google.com/uc?id=1IQIaKwQ61RQCDAQuNmLHh0fqzRw8wEVT", "checkpoint.tar.gz")
    download_and_extract("https://drive.google.com/uc?id=11pbvvFJLAS7qLv68odP80Ea6CYkjlQGi", "saved_model.tar.gz")

    print("Downloading voc.names...")
    download_file("https://raw.githubusercontent.com/pjreddie/darknet/master/data/voc.names",
                  Path("./data/classes/voc.names"))

    print("Converting TF graph...")
    shutil.rmtree('./saved_model', ignore_errors=True)
    convert_graph_def_to_saved_model('./saved_model',
                                     './yolov3_nano_416.pb',
                                     'input/input_data',
                                     ['pred_sbbox/concat_2:0', 'pred_mbbox/concat_2:0', 'pred_lbbox/concat_2:0'])

    print("Downloading sample video...")
    download_file("https://github.com/YunYang1994/tensorflow-yolov3/raw/master/docs/images/road.mp4", Path("road.mp4"))

    print("Files ready.")


if __name__ == "__main__":
    download_assets()
