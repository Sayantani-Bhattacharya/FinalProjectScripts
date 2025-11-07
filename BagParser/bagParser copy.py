# USED: to convert rosbag2 data to images and extract AprilTag poses
# this generates frame_output which then is given as input to the frameReader script to create a video.
# The generated video is now currently used as input to the AprilTag scripts file to output the tag pose, output video with bounding box, and csv. 
# This can be simplified to directly extract the tag poses from the rosbag2 data.


# To Run:   pyenv shell realsense-env  || python bagParser.py

from rosbags.highlevel import AnyReader
# from rosbags.image import message_to_cvimage
from rosbags.typesys import get_types_from_msg
from pathlib import Path
import cv2
import numpy as np
import csv, os
import apriltag
from scipy.spatial.transform import Rotation as R

# ---------- CONFIG ----------# for external camera
# bag_path = "/home/sayantani/Documents/Spring/FinalProj/ws/Sealbot/seal_data_manager/rosbag2_2025_10_30-23_14_45"  # for internal camera 
bag_path = "/home/sayantani/rosbag2_2025_10_30-23_14_13"  # for external camera 



topic_name = "/camera/camera/color/image_raw"   # for external camera
# topic_name = "/camera/image" # for internal camera 

output_dir = "frames_output"
csv_filename = "timestamps.csv"
tag_csv_filename = "tag_paths.csv"

fx, fy, cx, cy = 600, 600, 320, 240
tag_size = 0.16

os.makedirs(output_dir, exist_ok=True)


def message_to_cvimage(msg):
    """Convert sensor_msgs/msg/Image to OpenCV image (NumPy array)."""
    dtype_map = {
        'rgb8': np.uint8,
        'bgr8': np.uint8,
        'mono8': np.uint8,
    }

    if msg.encoding not in dtype_map:
        raise ValueError(f"Unsupported image encoding: {msg.encoding}")

    img = np.frombuffer(msg.data, dtype=dtype_map[msg.encoding]).reshape(msg.height, msg.width, -1)

    if msg.encoding == 'rgb8':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img

# AprilTag detector setup
options = apriltag.DetectorOptions(families='tag16h5')
detector = apriltag.Detector(options)

# ---------- READ ROSBAG ----------
with AnyReader([Path(bag_path)]) as reader:
    # Optional: inspect topics
    print("Topics found:", [x.topic for x in reader.connections])

    # Find the desired topic connection
    conn = next(c for c in reader.connections if c.topic == topic_name)

    with open(os.path.join(output_dir, csv_filename), "w", newline="") as csvfile, \
         open(os.path.join(output_dir, tag_csv_filename), "w", newline="") as tag_csvfile:

        writer = csv.writer(csvfile)
        tag_writer = csv.writer(tag_csvfile)
        writer.writerow(["frame_id", "timestamp_ns", "timestamp_sec", "image_file"])
        tag_writer.writerow(['tag_id', 'frame_id', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw'])

        tag_paths = {}
        frame_id = 0

        for connection, timestamp, rawdata in reader.messages(connections=[conn]):
            msg = reader.deserialize(rawdata, connection.msgtype)

            # Convert ROS Image → OpenCV
            cv_img = message_to_cvimage(msg).copy()
            if cv_img.ndim == 3 and cv_img.shape[2] == 4:
                cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGRA2BGR)

            img_file = f"frame_{frame_id:06d}.jpg"
            cv2.imwrite(os.path.join(output_dir, img_file), cv_img)

            timestamp_ns = timestamp
            timestamp_sec = timestamp_ns / 1e9
            writer.writerow([frame_id, int(timestamp_ns), f"{timestamp_sec:.6f}", img_file])

            # AprilTag detection
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            results = detector.detect(gray)

            for r in results:
                tag_id = r.tag_id
                M, init_error, final_error = detector.detection_pose(r, [fx, fy, cx, cy], tag_size)
                t = M[:3, 3]
                R_mat = M[:3, :3]
                quat = R.from_matrix(R_mat).as_quat()

                pose = (frame_id, t[0], t[1], t[2], quat[0], quat[1], quat[2], quat[3])
                tag_paths.setdefault(tag_id, []).append(pose)
                tag_writer.writerow([tag_id] + list(pose))

                # Draw detections
                (ptA, ptB, ptC, ptD) = r.corners
                pts = [tuple(map(int, p)) for p in (ptA, ptB, ptC, ptD)]
                cv2.polylines(cv_img, [np.array(pts, np.int32)], True, (0, 255, 0), 2)
                cX, cY = map(int, r.center)
                cv2.circle(cv_img, (cX, cY), 5, (0, 0, 255), -1)
                cv2.putText(cv_img, f"ID: {tag_id}", (pts[0][0], pts[0][1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            frame_id += 1

print(f"Extracted {frame_id} frames to {output_dir}/")
print(f"Timestamps → {csv_filename}")
print(f"AprilTag poses → {tag_csv_filename}")
