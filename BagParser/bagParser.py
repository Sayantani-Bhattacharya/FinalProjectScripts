# USED: to convert rosbag2 data to images and extract AprilTag poses
# this generates a video output with AprilTag detections and saves tag poses to a CSV file.
# To Run:   pyenv shell realsense-env  || python bagParser.py

from rosbags.highlevel import AnyReader
from pathlib import Path
import cv2
import numpy as np
import csv, os
import apriltag
from scipy.spatial.transform import Rotation as R

#######################################################################################
#                                  CONFIG
#######################################################################################

bag_path = "/home/sayantani/rosbag2_2025_11_06-23_19_11" 
output_dir = "frames_output"
csv_filename_Ext = "timestamps_ext.csv"
csv_filename_Int = "timestamps_int.csv"
tag_csv_filename_Ext = "tag_paths_ext.csv"
tag_csv_filename_Int = "tag_paths_int.csv"
video_filename_Ext = "output_video_ext.avi" 
video_filename_Int = "output_video_int.avi" 

os.makedirs(output_dir, exist_ok=True)

############################### External Camera #######################################
topic_name = "/camera/camera/color/image_raw"   
# AprilTag detector setup
options = apriltag.DetectorOptions(families='tag36h11') #tag16h5: new  || tag36h11: og  
detector = apriltag.Detector(options)
fx, fy, cx, cy = 600, 600, 320, 240
tag_size = 0.16

############################### Internal Camera #######################################
topic_nameInt = "/camera/image"   
# AprilTag detector setup
optionsInt = apriltag.DetectorOptions(families='tag16h5') #tag16h5: new  || tag36h11: og  
detectorInt = apriltag.Detector(options)
fxInt, fyInt, cxInt, cyInt = 1806.68775, 1801.14087, 1882.18218, 1404.07528   # With OpenCV Calib Matrix. TODO: verify std units here same as openCV.
tag_sizeInt = 0.007725     # The black square width ≈ 0.75 × 10.3 mm = 7.725 mm.   || 10.3 mm is with border



#######################################################################################
#                                  Helper Function
#######################################################################################

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

#######################################################################################
#                                  Read ROSBAG
#######################################################################################

with AnyReader([Path(bag_path)]) as reader:
    # Optional: inspect topics
    print("Topics found:", [x.topic for x in reader.connections])

    # Find the desired topic connection
    conn = next(c for c in reader.connections if c.topic == topic_name)
    connInt = next(c for c in reader.connections if c.topic == topic_nameInt)


    # Initialize video writers separately for external and internal cameras
    video_writer_ext = None
    video_writer_int = None

    with open(os.path.join(output_dir, csv_filename_Ext), "w", newline="") as csvfile, \
         open(os.path.join(output_dir, tag_csv_filename_Ext), "w", newline="") as tag_csvfile, \
            open(os.path.join(output_dir, csv_filename_Int), "w", newline="") as csvfileInt, \
            open(os.path.join(output_dir, tag_csv_filename_Int), "w", newline="") as tag_csvfileInt:

        #######################################################################################
        # External Tag Detection Loop

        writer_int = csv.writer(csvfile)
        tag_writer_int = csv.writer(tag_csvfile)
        writer_int.writerow(["frame_id", "timestamp_ns", "timestamp_sec"])
        tag_writer_int.writerow(['tag_id', 'frame_id', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw'])

        tag_paths = {}
        frame_id = 0

        for connection, timestamp, rawdata in reader.messages(connections=[conn]):
            msg = reader.deserialize(rawdata, connection.msgtype)

            # Convert ROS Image → OpenCV
            cv_img = message_to_cvimage(msg).copy()
            if cv_img.ndim == 3 and cv_img.shape[2] == 4:
                cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGRA2BGR)

            # Initialize video writer with frame dimensions and FPS
            if video_writer_ext is None:
                height, width = cv_img.shape[:2]
                video_writer_ext = cv2.VideoWriter(
                    os.path.join(output_dir, video_filename_Ext),
                    cv2.VideoWriter_fourcc(*'XVID'),
                    30,  # Assuming 30 FPS
                    (width, height)
                )

            timestamp_ns = timestamp
            timestamp_sec = timestamp_ns / 1e9
            writer_int.writerow([frame_id, int(timestamp_ns), f"{timestamp_sec:.6f}"])

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
                tag_writer_int.writerow([tag_id] + list(pose))

                # Draw detections
                (ptA, ptB, ptC, ptD) = r.corners
                pts = [tuple(map(int, p)) for p in (ptA, ptB, ptC, ptD)]
                cv2.polylines(cv_img, [np.array(pts, np.int32)], True, (0, 255, 0), 2)
                cX, cY = map(int, r.center)
                cv2.circle(cv_img, (cX, cY), 5, (0, 0, 255), -1)
                cv2.putText(cv_img, f"ID: {tag_id}", (pts[0][0], pts[0][1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Write the frame to the video
            video_writer_ext.write(cv_img)

            frame_id += 1

        print(f"Extracted {frame_id} frames from External Camera.")
        
        #######################################################################################
        # Internal Tag Detection Loop
        
        writer_ext = csv.writer(csvfile)
        tag_writer_ext = csv.writer(tag_csvfile)
        writer_ext.writerow(["frame_id", "timestamp_ns", "timestamp_sec"])
        tag_writer_ext.writerow(['tag_id', 'frame_id', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw'])

        tag_paths = {}
        frame_id = 0      
        
        for connection, timestamp, rawdata in reader.messages(connections=[connInt]):
            msg = reader.deserialize(rawdata, connection.msgtype)

            # Convert ROS Image → OpenCV
            cv_img = message_to_cvimage(msg).copy()
            if cv_img.ndim == 3 and cv_img.shape[2] == 4:
                cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGRA2BGR)
            
            if video_writer_int is None:
                height, width = cv_img.shape[:2]
                video_writer_int = cv2.VideoWriter(
                    os.path.join(output_dir, video_filename_Int),
                    cv2.VideoWriter_fourcc(*'XVID'),
                    30,  # Assuming 30 FPS
                    (width, height)
                )

            timestamp_ns = timestamp
            timestamp_sec = timestamp_ns / 1e9
            writer_ext.writerow([frame_id, int(timestamp_ns), f"{timestamp_sec:.6f}"])

            # AprilTag detection
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            results = detectorInt.detect(gray)

            for r in results:
                tag_id = r.tag_id
                M, init_error, final_error = detectorInt.detection_pose(r, [fxInt, fyInt, cxInt, cyInt], tag_sizeInt)
                t = M[:3, 3]
                R_mat = M[:3, :3]
                quat = R.from_matrix(R_mat).as_quat()

                pose = (frame_id, t[0], t[1], t[2], quat[0], quat[1], quat[2], quat[3])
                tag_paths.setdefault(tag_id, []).append(pose)
                tag_writer_ext.writerow([tag_id] + list(pose))

                # Draw detections
                (ptA, ptB, ptC, ptD) = r.corners
                pts = [tuple(map(int, p)) for p in (ptA, ptB, ptC, ptD)]
                cv2.polylines(cv_img, [np.array(pts, np.int32)], True, (255, 0, 0), 2)
                cX, cY = map(int, r.center)
                cv2.circle(cv_img, (cX, cY), 5, (0, 0, 255), -1)
                cv2.putText(cv_img, f"ID: {tag_id}", (pts[0][0], pts[0][1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            video_writer_int.write(cv_img)
            video_writer_int.write(cv_img)
            frame_id += 1

        print(f"Extracted {frame_id} frames from Internal Camera.")

    if video_writer_ext is not None:
        video_writer_ext.release()

    if video_writer_int is not None:
        video_writer_int.release()

#######################################################################################
#                   Relative Pose wrt to Static Tag
#######################################################################################


#######################################################################################
#                  Plot the Fused Centroid Path of External Tag
#######################################################################################


#######################################################################################
#          Plot the Relative Tag Pose for all Internal Tags wrt Timestamp
#######################################################################################

#######################################################################################
#                           Telemetry Data Plot
#######################################################################################


#######################################################################################
#                                  Completion Message
#######################################################################################

print(f"Extracted {frame_id} frames and saved video to {os.path.join(output_dir, video_filename_Ext)}")
print(f"Extracted {frame_id} frames and saved video to {os.path.join(output_dir, video_filename_Int)}")

print(f"Timestamps → {csv_filename_Ext}")
print(f"AprilTag poses ext → {tag_csv_filename_Ext}")
print(f"Timestamps → {csv_filename_Int}")
print(f"AprilTag poses int → {tag_csv_filename_Int}")


#######################################################################################
#                                  END OF FILE
#######################################################################################