#!/usr/bin/env python
# coding=utf-8

"""
Video Demonstration with YOLOv3 Model.

This script processes a video file to detect objects using a YOLOv3 model.
Users can specify the input and output video files via command line arguments.
"""

import os
import cv2
import time
import numpy as np
import tensorflow as tf
import argparse
import yolodemo.core.utils as utils


TF_CPP_MIN_LOG_LEVEL = '3'
TF_LOGGING_VERBOSITY = tf.compat.v1.logging.ERROR
MODEL_PATH = "./yolov3_nano_416.pb"
NUM_CLASSES = 20
INPUT_SIZE = 416
CODEC = 'mp4v'
FRAME_RATE = 20.0
RESOLUTION = (1280, 720)

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = TF_CPP_MIN_LOG_LEVEL
tf.compat.v1.logging.set_verbosity(TF_LOGGING_VERBOSITY)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="YOLOv3 Video Processing")
    parser.add_argument("--input", help="Path to the input video file", default="road.mp4")
    parser.add_argument("--output", help="Path to the output video file", default="out.mp4")
    return parser.parse_args()


def download_assets_if_needed(pb_file_path):
    """Downloads model assets if they are not present."""
    if not os.path.exists(pb_file_path):
        print("Model assets not found, downloading...")
        from yolodemo.download_assets import download_assets
        download_assets()


def setup_video_writer(output_path, codec=CODEC, frame_rate=FRAME_RATE, resolution=RESOLUTION):
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
    image_data = utils.image_preporcess(np.copy(frame_rgb), [input_size, input_size])
    image_data = image_data[np.newaxis, ...]

    pred_sbbox, pred_lbbox = session.run([return_tensors[1], return_tensors[2]], feed_dict={return_tensors[0]: image_data})
    pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)), np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)

    bboxes = utils.postprocess_boxes(pred_bbox, frame.shape[:2], input_size, 0.3)
    bboxes = utils.nms(bboxes, 0.45, method='nms')
    processed_frame = utils.draw_bbox(frame_rgb, bboxes)

    processed_frame_bgr = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
    return processed_frame_bgr, time.time() - start_time


def main():
    args = parse_arguments()
    pb_file = MODEL_PATH
    num_classes = NUM_CLASSES
    input_size = INPUT_SIZE
    graph = tf.Graph()

    download_assets_if_needed(pb_file)

    return_tensors = utils.read_pb_return_tensors(graph, pb_file, ["input/input_data:0", "pred_sbbox/concat_2:0", "pred_lbbox/concat_2:0"])
    process_video(args.input, args.output, graph, return_tensors)


if __name__ == "__main__":
    main()
    