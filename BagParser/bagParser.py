# rosbag2_2025_05_22-23_41_49_0.mcap   : bag name 
import rosbag2_py
import cv2
import numpy as np
from sensor_msgs.msg import Image
from rclpy.serialization import deserialize_message
from cv_bridge import CvBridge
import csv
import os
import apriltag
from scipy.spatial.transform import Rotation as R

# ---------- CONFIG ----------
bag_path = "/home/sayantani/Documents/Spring/FinalProj/ws/rosbag2_2025_10_30-22_48_53"  # path containing metadata.yaml
topic_name = "/camera/camera/color/image_raw"
output_dir = "frames_output"
csv_filename = "timestamps.csv" 
tag_csv_filename = "tag_paths.csv"

# Camera intrinsics (fx, fy, cx, cy) and tag size (in meters)
fx, fy, cx, cy = 600, 600, 320, 240  # Replace with your actual values
tag_size = 0.16  # Replace with your actual tag size

os.makedirs(output_dir, exist_ok=True)
bridge = CvBridge()

# Set up AprilTag detector
options = apriltag.DetectorOptions(families='tag16h5') #tag16h5 #tag36h11: og
detector = apriltag.Detector(options)

# ---------- SETUP READER ----------
storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id="mcap")
converter_options = rosbag2_py.ConverterOptions(
    input_serialization_format="cdr",
    output_serialization_format="cdr",
)

reader = rosbag2_py.SequentialReader()
reader.open(storage_options, converter_options)

# ---------- LOOP THROUGH MESSAGES ----------
tag_paths = {}  # {tag_id: [(frame_id, tx, ty, tz, qx, qy, qz, qw), ...]}
frame_id = 0

with open(os.path.join(output_dir, csv_filename), "w", newline="") as csvfile, \
     open(os.path.join(output_dir, tag_csv_filename), "w", newline="") as tag_csvfile:
    writer = csv.writer(csvfile)
    tag_writer = csv.writer(tag_csvfile)
    writer.writerow(["frame_id", "timestamp_ns", "timestamp_sec", "image_file"])
    tag_writer.writerow(['tag_id', 'frame_id', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw'])

    while reader.has_next():
        (topic, data, t) = reader.read_next()
        if topic != topic_name:
            continue

        # Deserialize and convert ROS2 Image message to OpenCV image
        img_msg = deserialize_message(data, Image)
        cv_img = bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")

        # Save frame
        img_file = f"frame_{frame_id:06d}.jpg"
        cv2.imwrite(os.path.join(output_dir, img_file), cv_img)

        # ROS2 timestamps are in nanoseconds
        timestamp_ns = img_msg.header.stamp.sec * 1e9 + img_msg.header.stamp.nanosec
        timestamp_sec = timestamp_ns / 1e9

        writer.writerow([frame_id, int(timestamp_ns), f"{timestamp_sec:.6f}", img_file])

        # Convert to grayscale for AprilTag detection
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        results = detector.detect(gray)

        for r in results:
            tag_id = r.tag_id
            M, init_error, final_error = detector.detection_pose(r, [fx, fy, cx, cy], tag_size)
            t = M[:3, 3]  # translation vector
            R_mat = M[:3, :3]  # rotation matrix
            quat = R.from_matrix(R_mat).as_quat()  # (x, y, z, w)
            pose = (frame_id, t[0], t[1], t[2], quat[0], quat[1], quat[2], quat[3])
            tag_paths.setdefault(tag_id, []).append(pose)
            tag_writer.writerow([tag_id] + list(pose))

            # Draw detections on the frame
            (ptA, ptB, ptC, ptD) = r.corners
            ptA = tuple(map(int, ptA))
            ptB = tuple(map(int, ptB))
            ptC = tuple(map(int, ptC))
            ptD = tuple(map(int, ptD))
            cX, cY = map(int, r.center)
            cv2.line(cv_img, ptA, ptB, (0, 255, 0), 2)
            cv2.line(cv_img, ptB, ptC, (0, 255, 0), 2)
            cv2.line(cv_img, ptC, ptD, (0, 255, 0), 2)
            cv2.line(cv_img, ptD, ptA, (0, 255, 0), 2)
            cv2.circle(cv_img, (cX, cY), 5, (0, 0, 255), -1)
            cv2.putText(cv_img, f"ID: {tag_id}", (ptA[0], ptA[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        frame_id += 1

print(f"Extracted {frame_id} frames to {output_dir}/ and timestamps to {csv_filename}")
print(f"Saved AprilTag paths to {tag_csv_filename}")
