# rosbag2_2025_05_22-23_41_49_0.mcap   : bag name 

import rosbag2_py
import cv2
import numpy as np
from sensor_msgs.msg import Image
from rclpy.serialization import deserialize_message
from cv_bridge import CvBridge
import csv
import os

# ---------- CONFIG ----------
bag_path = "/home/sayantani/Documents/Spring/FinalProj/scripts/BagParser/rosbag2_2025_05_22-23_41_49"  # path containing metadata.yaml
topic_name = "/camera/camera/color/image_raw"
output_dir = "frames_output"
csv_filename = "timestamps.csv"

os.makedirs(output_dir, exist_ok=True)
bridge = CvBridge()

# ---------- SETUP READER ----------
storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id="mcap")
converter_options = rosbag2_py.ConverterOptions(
    input_serialization_format="cdr",
    output_serialization_format="cdr",
)

reader = rosbag2_py.SequentialReader()
reader.open(storage_options, converter_options)

# ---------- LOOP THROUGH MESSAGES ----------
with open(os.path.join(output_dir, csv_filename), "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["frame_id", "timestamp_ns", "timestamp_sec", "image_file"])

    frame_id = 0
    while reader.has_next():
        (topic, data, t) = reader.read_next()
        if topic != topic_name:
            continue

        # img_msg = Image()
        img_msg = deserialize_message(data, Image)
        cv_img = bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")

        # Save frame
        img_file = f"frame_{frame_id:06d}.jpg"
        cv2.imwrite(os.path.join(output_dir, img_file), cv_img)

        # ROS2 timestamps are in nanoseconds
        timestamp_ns = img_msg.header.stamp.sec * 1e9 + img_msg.header.stamp.nanosec
        timestamp_sec = timestamp_ns / 1e9

        writer.writerow([frame_id, int(timestamp_ns), f"{timestamp_sec:.6f}", img_file])
        frame_id += 1

print(f" Extracted {frame_id} frames to {output_dir}/ and timestamps to {csv_filename}")
