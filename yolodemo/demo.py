#!/usr/bin/env python
# coding=utf-8

"""
Video Demonstration with YOLOv3 Model.

This script processes a video file to detect objects using a YOLOv3 model.
Users can specify the input and output video files via command line arguments.
"""

import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
import yolodemo.core.utils as utils

TF_CPP_MIN_LOG_LEVEL = "3"
TF_LOGGING_VERBOSITY = tf.compat.v1.logging.ERROR
script_dir = Path(__file__).parent
print(script_dir.resolve())
# Define the model path relative to the script directory
MODEL_PATH = str(script_dir / "assets" / "yolov3_nano_416.pb")
NUM_CLASSES = 20
INPUT_SIZE = 416
CODEC = "mp4v"
FRAME_RATE = 20.0
RESOLUTION = (1280, 720)

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = TF_CPP_MIN_LOG_LEVEL
tf.compat.v1.logging.set_verbosity(TF_LOGGING_VERBOSITY)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="YOLOv3 Video Processing")
    parser.add_argument(
        "--input", help="Path to the input video file", default="road.mp4"
    )
    parser.add_argument(
        "--output", help="Path to the output video file", default="out.mp4"
    )
    return parser.parse_args()


def download_assets_if_needed(pb_file_path):
    """Downloads model assets if they are not present."""
    if not os.path.exists(pb_file_path):
        print("Model assets not found, downloading...")
        from yolodemo.download_assets import download_assets

        download_assets()


def setup_video_writer(
    output_path, codec=CODEC, frame_rate=FRAME_RATE, resolution=RESOLUTION
):
    """Sets up the video writer."""
    fourcc = cv2.VideoWriter_fourcc(*codec)
    return cv2.VideoWriter(output_path, fourcc, frame_rate, resolution)


def process_video(video_path, output_path, graph, return_tensors):
    """Processes the video and displays the result."""
    try:
        cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
        display_enabled = True
    except cv2.error:
        print("Display not supported. Output will not be shown.")
        display_enabled = False
    print("Started processing", video_path)
    with tf.compat.v1.Session(graph=graph) as sess:
        vid = cv2.VideoCapture(video_path)
        out = setup_video_writer(output_path)
        while True:
            success, frame = vid.read()
            if not success:
                break

            processed_frame, _ = process_frame(frame, sess, return_tensors)
            out.write(processed_frame)
            if display_enabled:
                cv2.imshow("result", processed_frame)
    print("Saved processed video as", output_path)
    out.release()
    if display_enabled:
        cv2.destroyAllWindows()


def process_frame(frame, session, return_tensors):
    """Processes a single frame."""
    start_time = time.time()

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_data = utils.image_preporcess(np.copy(frame_rgb), [INPUT_SIZE, INPUT_SIZE])
    image_data = image_data[np.newaxis, ...]

    pred_sbbox, pred_lbbox = session.run(
        [return_tensors[1], return_tensors[2]],
        feed_dict={return_tensors[0]: image_data},
    )
    pred_bbox = np.concatenate(
        [
            np.reshape(pred_sbbox, (-1, 5 + NUM_CLASSES)),
            np.reshape(pred_lbbox, (-1, 5 + NUM_CLASSES)),
        ],
        axis=0,
    )

    bboxes = utils.postprocess_boxes(pred_bbox, frame.shape[:2], INPUT_SIZE, 0.3)
    bboxes = utils.nms(bboxes, 0.45, method="nms")
    processed_frame = utils.draw_bbox(frame_rgb, bboxes)

    processed_frame_bgr = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
    return processed_frame_bgr, time.time() - start_time


def run_detection_single_image(image_path):
    """
    Run object detection on a single image and display the result.

    Args:
    - image_path (str): Path to the image file.
    """
    # Ensure the model assets are available
    download_assets_if_needed(MODEL_PATH)

    # Load the TensorFlow graph
    graph = tf.Graph()
    with graph.as_default():
        return_tensors = utils.read_pb_return_tensors(
            graph,
            MODEL_PATH,
            ["input/input_data:0", "pred_sbbox/concat_2:0", "pred_lbbox/concat_2:0"],
        )

    # Start a TensorFlow session for inference
    with tf.compat.v1.Session(graph=graph) as sess:
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Image not found at {image_path}")

        # Process the image for YOLOv3 (resize, normalize, etc.)
        frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_data = utils.image_preporcess(
            np.copy(frame_rgb), [INPUT_SIZE, INPUT_SIZE]
        )
        image_data = image_data[np.newaxis, ...]  # Add batch dimension

        # Run detection
        pred_sbbox, pred_lbbox = sess.run(
            [return_tensors[1], return_tensors[2]],
            feed_dict={return_tensors[0]: image_data},
        )
        pred_bbox = np.concatenate(
            [
                np.reshape(pred_sbbox, (-1, 5 + NUM_CLASSES)),
                np.reshape(pred_lbbox, (-1, 5 + NUM_CLASSES)),
            ],
            axis=0,
        )

        # Post-process the detections (non-max suppression, etc.)
        bboxes = utils.postprocess_boxes(pred_bbox, image.shape[:2], INPUT_SIZE, 0.3)
        bboxes = utils.nms(bboxes, 0.45, method="nms")

    return bboxes


def extract_frame_from_video(video_path, frame_path, frame_number=0):
    """
    Extracts a frame from a given video file.

    Args:
    - video_path (str): Path to the video file.
    - frame_path (str): Path where the extracted frame will be saved.
    - frame_number (int): The number of the frame to extract.
    """
    vidcap = cv2.VideoCapture(video_path)
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    success, image = vidcap.read()
    if success:
        cv2.imwrite(frame_path, image)
    else:
        raise Exception("Failed to extract frame from video.")
    vidcap.release()


def main():
    args = parse_arguments()
    pb_file = MODEL_PATH
    graph = tf.Graph()

    download_assets_if_needed(pb_file)

    return_tensors = utils.read_pb_return_tensors(
        graph,
        pb_file,
        ["input/input_data:0", "pred_sbbox/concat_2:0", "pred_lbbox/concat_2:0"],
    )
    process_video(args.input, args.output, graph, return_tensors)


if __name__ == "__main__":
    main()
