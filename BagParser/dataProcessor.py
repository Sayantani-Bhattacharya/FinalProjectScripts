# This file is intended to be used after the raw data is extracted from the bags by bagPaeser. 
# And we process the pose data of both internal and external camera sensors here to be ready for model training.

import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

#######################################################################################
#                         Configuration Parameters
#######################################################################################

# External Cube Dimensions
cube_size = 0.166  # meters   
static_tag_id = 4  # Mid of the wisker arrays in seal head.
tag_rel_poses = {}  

#######################################################################################
#               Int Tag Thresholding to prevent false positives
#######################################################################################

# Input is csv file and output is filtered tag_paths: but what about the frames where no tags are visible?: checksum sort logic in the perception model pipeline. 
# TODO: Edge Cases

# Tag pose reference with quaternion fixed: Calculated from the static_ref script.
internal_tag_id_rel_pose_map = {
    0: {"position": [0.01431607, 0.04001835, -0.07066817], "quaternion": [0.7071, 0.0, 0.7071, 0.0]},
    1: {"position": [0.00856013, 0.03377553, -0.04614408], "quaternion": [0.7071, 0.0, 0.7071, 0.0]},
    2: {"position": [0.01684475, 0.03069469, -0.01257934], "quaternion": [0.7071, 0.0, 0.7071, 0.0]},
    3: {"position": [-0.00075084, -0.05271414, -0.01421607], "quaternion": [0.09239, 0.3827, 0.0, 0.0]},
    5: {"position": [-0.01197729, -0.04446193, -0.0077042], "quaternion": [0.09239, 0.3827, 0.0, 0.0]},
    6: {"position": [-0.0211849, -0.03541196, -0.00963364], "quaternion": [0.09239, 0.3827, 0.0, 0.0]},
    7: {"position": [-0.01306488, -0.03531467, 0.01683272], "quaternion": [0.09239, 0.3827, 0.0, 0.0]},
    8: {"position": [-0.02081204, -0.05283257, -0.03941675], "quaternion": [0.09239, 0.3827, 0.0, 0.0]},
    9: {"position": [-0.01011581, -0.05187832, -0.03616606], "quaternion": [0.09239, 0.3827, 0.0, 0.0]},
    10: {"position": [-0.00087129, -0.04338111, 0.00374754], "quaternion": [0.09239, 0.3827, 0.0, 0.0]},
    11: {"position": [0.00564222, 0.02567888, -0.03249634], "quaternion": [0.7071, 0.0, 0.7071, 0.0]},
    12: {"position": [0.02347466, 0.04053535, -0.07390323], "quaternion": [0.7071, 0.0, 0.7071, 0.0]},
    13: {"position": [-0.02242572, -0.04503598, -0.01635032], "quaternion": [0.09239, 0.3827, 0.0, 0.0]},
    14: {"position": [0.01984042, 0.03416312, -0.05061772], "quaternion": [0.7071, 0.0, 0.7071, 0.0]},
    16: {"position": [-0.00189823, 0.03547269, -0.03190359], "quaternion": [0.7071, 0.0, 0.7071, 0.0]},
    17: {"position": [-0.00726191, 0.02756754, -0.01221053], "quaternion": [0.7071, 0.0, 0.7071, 0.0]},
    18: {"position": [0.00206168, 0.03940511, -0.06486038], "quaternion": [0.7071, 0.0, 0.7071, 0.0]}
}

margin_error_position = 0.06 # 6 cm

def unfilter_rel_paths():
    # Read CSV and group by tag_id
    tag_poses = {}
    with open('frames_output_Cut_PT4/tag_paths_int.csv', 'r') as f: 
        reader = csv.DictReader(f)
        for row in reader:
            tag_id = int(row['tag_id'])
            frame_id = int(row['frame_id'])
            tx = float(row['tx'])
            ty = float(row['ty'])
            tz = float(row['tz'])
            qx = float(row['qx'])
            qy = float(row['qy'])
            qz = float(row['qz'])
            qw = float(row['qw'])

            if frame_id not in tag_poses:
                tag_poses[frame_id] = {}
            tag_poses[frame_id][tag_id] = (np.array([tx, ty, tz]), np.array([qx, qy, qz, qw]))

    rel_tag_poses = {}

    # For fixed frame_id, this will have all "19" poses and will iterate next to the next frame_id.
    for frame_id, poses in tag_poses.items():
        if static_tag_id not in poses: # this discards all frames where the static tag is not detected.
            continue

        static_pos, static_quaternion = poses[static_tag_id]
        static_rot = R.from_quat(static_quaternion)

        for tag_id, (pos, quat) in poses.items():
            if tag_id == static_tag_id:
                continue

            # Check if tag_id is in the reference map
            if tag_id not in internal_tag_id_rel_pose_map:
                continue

            rel_pos = static_rot.inv().apply(pos - static_pos)
            rot = R.from_quat(quat)
            rel_rot = static_rot.inv() * rot

            # Use rel_pos and rel_rot for further processing.
            if frame_id not in rel_tag_poses:
                rel_tag_poses[frame_id] = {}
            rel_tag_poses[frame_id][tag_id] = (rel_pos, rel_rot)
        
    return rel_tag_poses
   

def filter_false_positive():
    # Read CSV and group by tag_id
    tag_poses = {}
    with open('frames_output_Cut_PT4/tag_paths_int.csv', 'r') as f: 
        reader = csv.DictReader(f)
        for row in reader:
            tag_id = int(row['tag_id'])
            frame_id = int(row['frame_id'])
            tx = float(row['tx'])
            ty = float(row['ty'])
            tz = float(row['tz'])
            qx = float(row['qx'])
            qy = float(row['qy'])
            qz = float(row['qz'])
            qw = float(row['qw'])

            if frame_id not in tag_poses:
                tag_poses[frame_id] = {}
            tag_poses[frame_id][tag_id] = (np.array([tx, ty, tz]), np.array([qx, qy, qz, qw]))

    rel_tag_poses = {}

    # For fixed frame_id, this will have all "19" poses and will iterate next to the next frame_id.
    for frame_id, poses in tag_poses.items():
        if static_tag_id not in poses: # this discards all frames where the static tag is not detected.
            continue

        static_pos, static_quaternion = poses[static_tag_id]
        static_rot = R.from_quat(static_quaternion)

        for tag_id, (pos, quat) in poses.items():
            if tag_id == static_tag_id:
                continue

            # Check if tag_id is in the reference map
            if tag_id not in internal_tag_id_rel_pose_map:
                continue

            rel_pos = static_rot.inv().apply(pos - static_pos)
            rot = R.from_quat(quat)
            rel_rot = static_rot.inv() * rot

            # Check if the relative position is within margin error from the reference position.
            ref_pos = np.array(internal_tag_id_rel_pose_map[tag_id]["position"])
            if np.linalg.norm(rel_pos - ref_pos) > margin_error_position:
                continue

            # # Check if the relative rotation is within margin error from the reference quaternion.
            # ref_quat = np.array(internal_tag_id_rel_pose_map[tag_id]["quaternion"])
            # ref_rot = R.from_quat(ref_quat)
            # if rel_rot.inv() * ref_rot.magnitude() > margin_error_position:
            #     continue

            # Use rel_pos and rel_rot for further processing.
            if frame_id not in rel_tag_poses:
                rel_tag_poses[frame_id] = {}
            rel_tag_poses[frame_id][tag_id] = (rel_pos, rel_rot)
        
    return rel_tag_poses
            

#######################################################################################
#                   Relative Pose wrt to Static Tag
#######################################################################################

# # Relative poses global variable
# tx_rel = []
# ty_rel = []
# tz_rel = []

'''
Plot the 3D relative pose of all internal tags with respect to a static tag (ID = static_tag_id) over time.
'''
def plot_3d_relative_pose(ax_relative):
    ax_relative.clear()  # Clear the previous plot
    # Read CSV and group by tag_id
    tag_paths = {}
    with open('frames_output/tag_paths_int.csv', 'r') as f: 
        reader = csv.DictReader(f)
        for row in reader:
            tag_id = int(row['tag_id'])
            pose = (
                int(row['frame_id']),
                float(row['tx']),
                float(row['ty']),
                float(row['tz']),
                float(row['qx']),
                float(row['qy']),
                float(row['qz']),
                float(row['qw'])
            )
            tag_paths.setdefault(tag_id, []).append(pose)

    # Extract pose with ID = 0 as reference
    if static_tag_id not in tag_paths:
        print("No tag with ID static_tag_id found for reference.")
        return
    ref_poses = sorted(tag_paths[0], key=lambda x: x[0])  # sort by frame_idx

    # Plot each tag's path relative to tag 0: as in x-x1, y-y1, z-z1

    for tag_id, poses in tag_paths.items():
        if tag_id == static_tag_id:
            continue
        poses = sorted(poses, key=lambda x: x[0])
        tx_rel = []
        ty_rel = []
        tz_rel = []

        for p in poses:
            frame_idx = p[0]
            ref_pose = next((rp for rp in ref_poses if rp[0] == frame_idx), None)
            if ref_pose is None:
                continue

            # Extract translation poses
            t = np.array([p[1], p[2], p[3]])
            t_ref = np.array([ref_pose[1], ref_pose[2], ref_pose[3]])
            # Extract rotation poses
            R = R.from_quat(p[4:8]).as_matrix()
            R_ref = R.from_quat(ref_pose[4:8]).as_matrix()

            # Compute relative transformation
            R_rel =  R_ref.T @ R
            t_rel = R_ref.T @ (t - t_ref)

            tx_rel.append(t_rel[0])
            ty_rel.append(t_rel[1])
            tz_rel.append(t_rel[2])      
        
        ax_relative.plot(tx_rel, ty_rel, tz_rel, label=f'Tag {tag_id}')
    
    ax_relative.set_xlabel('X rel to Static Tag')
    ax_relative.set_ylabel('Y rel to Static Tag')
    ax_relative.set_zlabel('Z rel to Static Tag')
    ax_relative.legend()
    ax_relative.set_title('Relative 3D Poses to Static Tag')
    plt.show(block=False)
    plt.pause(0.01)

'''
Plot the 3D relative pose of a specific tag relative to Static Tag over time. 
'''
def plot_relative_pose_indv(tag_id, ax_indv):
    ax_indv.clear()  # Clear the previous plot
    tag_paths = {}
    with open('frames_output/tag_paths_int.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            t_id = int(row['tag_id'])
            pose = (
                int(row['frame_id']),
                float(row['tx']),
                float(row['ty']),
                float(row['tz']),
                float(row['qx']),
                float(row['qy']),
                float(row['qz']),
                float(row['qw'])
            )
            tag_paths.setdefault(t_id, []).append(pose)
    if static_tag_id not in tag_paths or tag_id not in tag_paths:
        print(f"Static Tag or Tag {tag_id} not found.")
        return
    ref_poses = sorted(tag_paths[0], key=lambda x: x[0])
    poses = sorted(tag_paths[tag_id], key=lambda x: x[0])
    tx_rel = []
    ty_rel = []
    tz_rel = []

    for p in poses:
        frame_idx = p[0]
        ref_pose = next((rp for rp in ref_poses if rp[0] == frame_idx), None)
        if ref_pose is None:
            continue

        # Extract translation poses
        t = np.array([p[1], p[2], p[3]])
        t_ref = np.array([ref_pose[1], ref_pose[2], ref_pose[3]])
        # Extract rotation poses
        R_stat = R.from_quat(p[4:8]).as_matrix()
        R_ref = R.from_quat(ref_pose[4:8]).as_matrix()

        # Compute relative transformation
        R_rel =  R_ref.T @ R_stat
        t_rel = R_ref.T @ (t - t_ref)

        tx_rel.append(t_rel[0])
        ty_rel.append(t_rel[1])
        tz_rel.append(t_rel[2])
    
    
    ax_indv.plot(tx_rel, ty_rel, tz_rel, label=f'Relative Path of Tag {tag_id}')
    ax_indv.set_xlabel('X rel to Static Tag')
    ax_indv.set_ylabel('Y rel to Static Tag')
    ax_indv.set_zlabel('Z rel to Static Tag')
    ax_indv.legend()
    ax_indv.set_title(f'Relative Path of Tag {tag_id} to Static Tag')
    plt.show()

'''
Plot the individual x, y, z signals of a specific tag separately with respect to a static tag (ID = static_tag_id) over time.
'''
def plot_idv(tag_id, ax_x = None, ax_y= None, ax_z= None):
    # ax_indv_axis.clear()  # Clear the previous plot
    tag_paths = {}
    with open('frames_output/tag_paths_int.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            t_id = int(row['tag_id'])
            pose = (
                int(row['frame_id']),
                float(row['tx']),
                float(row['ty']),
                float(row['tz']),
                float(row['qx']),
                float(row['qy']),
                float(row['qz']),
                float(row['qw'])
            )
            tag_paths.setdefault(t_id, []).append(pose)
    if tag_id not in tag_paths:
        print(f"Tag {tag_id} not found.")
        return

    poses = sorted(tag_paths[tag_id], key=lambda x: x[0])
    frame_idxs = [p[0] for p in poses]
    tx = [p[1] for p in poses]
    ty = [p[2] for p in poses]
    tz = [p[3] for p in poses]

    # Relative to static tag.
    if static_tag_id in tag_paths:
        static_poses = sorted(tag_paths[static_tag_id], key=lambda x: x[0])
        static_dict = {p[0]: p for p in static_poses}
        tx_rel = []
        ty_rel = []
        tz_rel = []
        for p in poses:
            frame_idx = p[0]
            if frame_idx in static_dict:
                static_p = static_dict[frame_idx]
                tx_rel.append(p[1] - static_p[1])
                ty_rel.append(p[2] - static_p[2])
                tz_rel.append(p[3] - static_p[3])
            else:
                tx_rel.append(p[1])
                ty_rel.append(p[2])
                tz_rel.append(p[3])
        tx, ty, tz = tx_rel, ty_rel, tz_rel

    # Plot X signal
    ax_x.plot(frame_idxs, tx, label='X', color='r')
    ax_x.set_xlabel('Frame Index')
    ax_x.set_ylabel('X Position (m)')
    ax_x.legend()
    ax_x.grid()

    # Plot Y signal
    ax_y.plot(frame_idxs, ty, label='Y', color='g')
    ax_y.set_xlabel('Frame Index')
    ax_y.set_ylabel('Y Position (m)')
    ax_y.legend()
    ax_y.grid()

    # Plot Z signal
    ax_z.plot(frame_idxs, tz, label='Z', color='b')
    ax_z.set_xlabel('Frame Index')
    ax_z.set_ylabel('Z Position (m)')
    ax_z.legend()
    ax_z.grid()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

'''
Plot the individual x, y, z signals of a specific tag relative to another specific tag over time.
'''
def plot_idv_comparative(tag_id_1, tag_id_2, ax_indv_axis, ax_x = None, ax_y= None, ax_z= None):
    # Plot the individual x, y, z signals of a specific tag separately over time.
    ax_indv_axis.clear()  # Clear the previous plot
    tag_paths = {}
    with open('tag_paths.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            t_id = int(row['tag_id'])
            pose = (
                int(row['frame_id']),
                float(row['tx']),
                float(row['ty']),
                float(row['tz']),
                float(row['qx']),
                float(row['qy']),
                float(row['qz']),
                float(row['qw'])
            )
            tag_paths.setdefault(t_id, []).append(pose)
    if tag_id_1 not in tag_paths or tag_id_2 not in tag_paths:
        print(f"Tag {tag_id_1} or Static Tag {tag_id_2} not found.")
        return
    tag_id1_poses = sorted(tag_paths[tag_id_1], key=lambda x: x[0])
    tag_id2_poses = sorted(tag_paths[tag_id_2], key=lambda x: x[0])

    frame_idxs = []
    tx_rel = []
    ty_rel = []
    tz_rel = []

    for p in tag_id1_poses:
        frame_idx = p[0]
        tag_id1_pose = next((sp for sp in tag_id1_poses if sp[0] == frame_idx), None)
        tag_id2_pose = next((sp for sp in tag_id2_poses if sp[0] == frame_idx), None)

        if tag_id2_pose is None:
            continue
        tag_1 = np.array([tag_id1_pose[1], tag_id1_pose[2], tag_id1_pose[3]])
        tag_2 = np.array([tag_id2_pose[1], tag_id2_pose[2], tag_id2_pose[3]])
        frame_idxs.append(frame_idx)
        tx_rel.append(tag_1[0] - tag_2[0])
        ty_rel.append(tag_1[1] - tag_2[1])
        tz_rel.append(tag_1[2] - tag_2[2])

    # Plot X signal
    ax_x.plot(frame_idxs, tx_rel, label='X rel to Static Tag', color='r')
    ax_x.set_xlabel('Frame Index')
    ax_x.set_ylabel('X Position (m)')
    ax_x.legend()
    ax_x.grid()

    # Plot Y signal
    ax_y.plot(frame_idxs, ty_rel, label='Y rel to Static Tag', color='g')
    ax_y.set_xlabel('Frame Index')
    ax_y.set_ylabel('Y Position (m)')
    ax_y.legend()
    ax_y.grid()

    # Plot Z signal
    ax_z.plot(frame_idxs, tz_rel, label='Z rel to Static Tag', color='b')
    ax_z.set_xlabel('Frame Index')
    ax_z.set_ylabel('Z Position (m)')
    ax_z.legend()
    ax_z.grid()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    # plt.pause(0.01)

'''
Plot the individual x, y, z signals of multiple specific filtered tags over time.
'''
def plot_filtered_tags(rel_tag_poses, ax_x, ax_y, ax_z, tagA=2, tagB=4, tagC=5, tagD=5):
    ax_x.clear()
    ax_y.clear()
    ax_z.clear()

    frame_idxs = sorted(rel_tag_poses.keys())
    tx_A, ty_A, tz_A = [], [], []
    tx_B, ty_B, tz_B = [], [], []
    tx_C, ty_C, tz_C = [], [], []
    tx_D, ty_D, tz_D = [], [], []

    default_value = 0

    for frame_idx in frame_idxs:
        poses = rel_tag_poses[frame_idx]
        if tagA in poses:
            pos_A, _ = poses[tagA]
            tx_A.append(pos_A[0])
            ty_A.append(pos_A[1])
            tz_A.append(pos_A[2])
        else:
            tx_A.append(default_value)
            ty_A.append(default_value)
            tz_A.append(default_value)
        if tagB in poses:
            pos_B, _ = poses[tagB]
            tx_B.append(pos_B[0])
            ty_B.append(pos_B[1])
            tz_B.append(pos_B[2])
        else:
            tx_B.append(default_value)
            ty_B.append(default_value)
            tz_B.append(default_value)
        if tagC in poses:
            pos_C, _ = poses[tagC]
            tx_C.append(pos_C[0])
            ty_C.append(pos_C[1])
            tz_C.append(pos_C[2])
        else:
            tx_C.append(default_value)
            ty_C.append(default_value)
            tz_C.append(default_value)
        if tagD in poses:
            pos_D, _ = poses[tagD]
            tx_D.append(pos_D[0])
            ty_D.append(pos_D[1])
            tz_D.append(pos_D[2])
        else:   
            tx_D.append(default_value)
            ty_D.append(default_value)
            tz_D.append(default_value)

    # Plot X: TODO: need to fix the lengths if some tags are missing in some frames.
    # Also separate each tag frame idxs plots.
    ax_x.plot(tx_A, label=f'Tag {tagA}', color='r')
    ax_x.plot(tx_B, label=f'Tag {tagB}', color='g')
    ax_x.plot(tx_C, label=f'Tag {tagC}', color='b')
    ax_x.plot(tx_D, label=f'Tag {tagD}', color='m')
    ax_x.set_xlabel('Frame Index')
    ax_x.set_ylabel('X Position (m)')
    ax_x.legend()
    ax_x.grid()

    # Plot Y
    ax_y.plot(ty_A, label=f'Tag {tagA}', color='r')
    ax_y.plot(ty_B, label=f'Tag {tagB}', color='g')
    ax_y.plot(ty_C, label=f'Tag {tagC}', color='b')
    ax_y.plot(ty_D, label=f'Tag {tagD}', color='m')
    ax_y.set_xlabel('Frame Index')
    ax_y.set_ylabel('Y Position (m)')
    ax_y.legend()
    ax_y.grid()

    # Plot Z
    ax_z.plot( tz_A, label=f'Tag {tagA}', color='r')
    ax_z.plot(tz_B, label=f'Tag {tagB}', color='g')
    ax_z.plot(tz_C, label=f'Tag {tagC}', color='b')
    ax_z.plot(tz_D, label=f'Tag {tagD}', color='m')
    ax_z.set_xlabel('Frame Index')
    ax_z.set_ylabel('Z Position (m)')
    ax_z.legend()
    ax_z.grid()
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


"""
    Calculate the relative pose from the start_pose to the current_pose.
    Both poses are given as (position, rotation) tuples.
"""
def calculate_rel_pose(current_pose, start_pose):
    
    curr_pos, curr_rot = current_pose
    start_pos, start_rot = start_pose

    # Calculate relative rotation
    rel_rot = start_rot.inv() * curr_rot

    # Calculate relative position
    rel_pos = start_rot.inv().apply(curr_pos - start_pos)

    return rel_pos

'''
Plot the individual x, y, z, roll, pitch, yaw signals of multiple specific filtered tags over time: Displcement from initial position.
'''
def plot_filtered_tags_displacement(rel_tag_poses, ax_x, ax_y, ax_z, tagA=None, tagB=None, tagC=None, tagD=None):
    ax_x.clear()
    ax_y.clear()
    ax_z.clear()

    frame_idxs = sorted(rel_tag_poses.keys())
    tx_A, ty_A, tz_A = [], [], []
    tx_B, ty_B, tz_B = [], [], []
    tx_C, ty_C, tz_C = [], [], []
    tx_D, ty_D, tz_D = [], [], []

    default_value = 0

    # Calculate the starting positions for displacement calculation
    # TODO: need to check if i want all pose relative to the first tag pose, or if I want pose relative to the previos frame pose. 
    # Right now I am doing relative to the first detected pose.

    # TODO: Assuming the first frame has all tags visible: Necessary for this to work.
    start_pos = {}
    poses = rel_tag_poses[frame_idxs[0]]
    if tagA in poses:
        # pos_A, rot_A = poses[tagA]
        start_pos[tagA] = poses[tagA]
    if tagB in poses:
        # pos_B, _ = poses[tagB]
        start_pos[tagB] = poses[tagB]
    if tagC in poses:
        # pos_C, _ = poses[tagC]
        start_pos[tagC] = poses[tagC]
    if tagD in poses:
        # pos_D, _ = poses[tagD]
        start_pos[tagD] = poses[tagD]

    for frame_idx in frame_idxs:
        poses = rel_tag_poses[frame_idx]
        if tagA in poses:
            pos_A = calculate_rel_pose(poses[tagA], start_pos[tagA])
            tx_A.append(pos_A[0])
            ty_A.append(pos_A[1])
            tz_A.append(pos_A[2])
        else:
            tx_A.append(default_value)
            ty_A.append(default_value)
            tz_A.append(default_value)
        if tagB in poses:
            pos_B = calculate_rel_pose(poses[tagB], start_pos[tagB])
            tx_B.append(pos_B[0])
            ty_B.append(pos_B[1])
            tz_B.append(pos_B[2])
        else:
            tx_B.append(default_value)
            ty_B.append(default_value)
            tz_B.append(default_value)
        if tagC in poses:
            pos_C = calculate_rel_pose(poses[tagC], start_pos[tagC])
            tx_C.append(pos_C[0])
            ty_C.append(pos_C[1])
            tz_C.append(pos_C[2])
        else:
            tx_C.append(default_value)
            ty_C.append(default_value)
            tz_C.append(default_value)
        if tagD in poses:
            pos_D = calculate_rel_pose(poses[tagD], start_pos[tagD])
            tx_D.append(pos_D[0])
            ty_D.append(pos_D[1])
            tz_D.append(pos_D[2])
        else:   
            tx_D.append(default_value)
            ty_D.append(default_value)
            tz_D.append(default_value)

    # Plot X: TODO: need to fix the lengths if some tags are missing in some frames.
    # Also separate each tag frame idxs plots.
    ax_x.plot(tx_A, label=f'Tag {tagA}', color='r')
    ax_x.plot(tx_B, label=f'Tag {tagB}', color='g')
    ax_x.plot(tx_C, label=f'Tag {tagC}', color='b')
    ax_x.plot(tx_D, label=f'Tag {tagD}', color='m')
    ax_x.set_xlabel('Frame Index')
    ax_x.set_ylabel('X Position (m)')
    ax_x.legend()
    ax_x.grid()

    # Plot Y
    ax_y.plot(ty_A, label=f'Tag {tagA}', color='r')
    ax_y.plot(ty_B, label=f'Tag {tagB}', color='g')
    ax_y.plot(ty_C, label=f'Tag {tagC}', color='b')
    ax_y.plot(ty_D, label=f'Tag {tagD}', color='m')
    ax_y.set_xlabel('Frame Index')
    ax_y.set_ylabel('Y Position (m)')
    ax_y.legend()
    ax_y.grid()

    # Plot Z
    ax_z.plot( tz_A, label=f'Tag {tagA}', color='r')
    ax_z.plot(tz_B, label=f'Tag {tagB}', color='g')
    ax_z.plot(tz_C, label=f'Tag {tagC}', color='b')
    ax_z.plot(tz_D, label=f'Tag {tagD}', color='m')
    ax_z.set_xlabel('Frame Index')
    ax_z.set_ylabel('Z Position (m)')
    ax_z.legend()
    ax_z.grid()
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


#######################################################################################
#                  Plot the Fused Centroid Path of External Tag
#######################################################################################

# Defining rigid transformation matrix from each id to centroid [ Assuming facce 1 as front]
# T_1C: 1 wrt Centroid (0,0,0)
# T_C1: Centroid (0,0,0) wrt 1
T_1C = np.array([[1, 0, 0, cube_size / 2],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]])

T_C1 = np.array([[1, 0, 0, -cube_size / 2],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]])

T_2C = np.array([[0, 0, 1, 0],
                [0, 1, 0, cube_size / 2],
                [-1, 0, 0, 0],
                [0, 0, 0, 1]])

T_C2 = np.array([[0, 0, 1, 0],
                [0, 1, 0, -cube_size / 2],
                [-1, 0, 0, 0],
                [0, 0, 0, 1]])

T_3C = np.array([[0, 0, 1, -cube_size / 2],
                [0, 1, 0, 0],
                [-1, 0, 0, 0],
                [0, 0, 0, 1]])

T_C3 = np.array([[0, 0, 1, cube_size / 2],
                [0, 1, 0, 0],
                [-1, 0, 0, 0],
                [0, 0, 0, 1]])

T_4C = np.array([[0, 0, 1, 0],
                [0, 1, 0, -cube_size / 2],
                [-1, 0, 0, 0],
                [0, 0, 0, 1]])

T_C4 = np.array([[0, 0, 1, 0],
                [0, 1, 0, cube_size / 2],
                [-1, 0, 0, 0],
                [0, 0, 0, 1]])

T_5C = np.array([[0, 0, 1, 0],
                [0, 1, 0, 0],
                [-1, 0, 0, cube_size / 2],
                [0, 0, 0, 1]])

T_C5 = np.array([[0, 0, 1, 0],
                [0, 1, 0, 0],
                [-1, 0, 0, - cube_size / 2],
                [0, 0, 0, 1]])

# Global variable to store fused centroid path.
fused_centroid_path = [] 
total_frames = 0

# Funtion to read tag_paths_ext.csv to get current frame poses.
def read_frame_poses():
    all_frame_poses = []
    with open('frames_output/tag_paths_ext.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            tag_id = int(row['tag_id'])
            pose = (
                int(row['frame_id']),
                float(row['tx']),
                float(row['ty']),
                float(row['tz']),
                float(row['qx']),
                float(row['qy']),
                float(row['qz']),
                float(row['qw'])
            )

            all_frame_poses.append((tag_id, pose[0] , pose[1], pose[2], pose[3], pose[4], pose[5], pose[6], pose[7]))

    # print("Read the Current Frame Poses from csv with length of: ", len(all_frame_poses))
    return all_frame_poses

def get_current_frame_poses(all_frame_poses, frame_idx):
    current_frame_poses = [p for p in all_frame_poses if p[1] == frame_idx]   ###  is 0 or 1
    return current_frame_poses

def pose_fusion(poses):  #frame_idx might be needed to be removed evrywhere in this function.
    """
    Function to fuse multiple tag poses to get a more stable pose estimate for the centroid: based on the tags visible.
    Input: list of centroid poses from all visible tags [(frame_idx, tx, ty, tz, qx, qy, qz, qw), ...]
    Output: fused pose (tx, ty, tz, qx, qy, qz, qw)
    """

    if not poses:
        return None

    # frame_idx = poses[0][0]
    translations = np.array([[p[0], p[1], p[2]] for p in poses])
    quaternions = np.array([[p[3], p[4], p[5], p[6]] for p in poses])

    # Average translation
    avg_translation = np.mean(translations, axis=0)

    # Average quaternion (using Singular Value Decomposition method): 
    # Its a least squares solution to find the average quaternion.
    # More robust than simple averaging.
    A = np.zeros((4, 4))
    for q in quaternions:
        A += np.outer(q, q)
    A /= len(quaternions)
    _, _, Vt = np.linalg.svd(A)
    avg_quaternion = Vt[0]

    return ( avg_translation[0], avg_translation[1], avg_translation[2],
            avg_quaternion[0], avg_quaternion[1], avg_quaternion[2], avg_quaternion[3])

# TODO: Can be later merged with plot_relative_pose_indv.
'''
Plot the 3D relative pose of a specific external tag with respect to a static tag (ID = static_tag_id).

TODO: this has logic error, for cases when the mentioned tag is not visible in some frames. 
''' 
def plot_relative_pose_indvExt(tag_id, ax_indv, all_frame_poses):
    ax_indv.clear()  # Clear the previous plot
    static_tag_id = 0 # Ext tag static id.
    tag_paths = {}
    tx_rel = []
    ty_rel = []
    tz_rel = []

    total_frames = max(p[0] for p in all_frame_poses) + 1  # Assuming frame indices start from 0

    for frame_idx in range(total_frames):
        # Collecting same frame index poses, all tags.
        current_frame_poses = get_current_frame_poses(all_frame_poses, frame_idx)


    for pose in all_frame_poses:

        t_id = int(pose[1])
        tag_paths.setdefault(t_id, []).append(pose) #this is woring the tag path need to be populated, and same time me dono tags ka relative hona chahiye.
        if static_tag_id not in tag_paths or tag_id not in tag_paths:
            print(f"Static Tag or Tag {tag_id} not found.")
            return
        ref_poses = sorted(tag_paths[static_tag_id], key=lambda x: x[0])
        poses = sorted(tag_paths[tag_id], key=lambda x: x[0])
        for p in poses:
            frame_idx = p[0]
            ref_pose = next((rp for rp in ref_poses if rp[0] == frame_idx), None)
            if ref_pose is None:
                continue

            # Extract translation poses
            t = np.array([p[2], p[3], p[4]])
            t_ref = np.array([ref_pose[2], ref_pose[3], ref_pose[4]])
            # Extract rotation poses
            R = R.from_quat(p[5:9]).as_matrix()
            R_ref = R.from_quat(ref_pose[5:9]).as_matrix()

            # Compute relative transformation
            R_rel =  R_ref.T @ R
            t_rel = R_ref.T @ (t - t_ref)

            tx_rel.append(t_rel[0])
            ty_rel.append(t_rel[1])
            tz_rel.append(t_rel[2])

    ax_indv.plot(tx_rel, ty_rel, tz_rel, label=f'Relative Path of Tag {tag_id}')
    ax_indv.set_xlabel('X rel to Static Tag')
    ax_indv.set_ylabel('Y rel to Static Tag')
    ax_indv.set_zlabel('Z rel to Static Tag')
    ax_indv.legend()
    ax_indv.set_title(f'Relative Path of Tag {tag_id} to Static Tag')
    plt.show()

'''
Plot the 3D centroid path of the cube relative to the static tag ID = 0 over all frames.
'''
def plot_centroid_path(ax_centroid, all_frame_poses, start_frame=0, end_frame=None, axis_equal=False):   
    # Clear the plot
    ax_centroid.clear()

    # Centroid poses from all tags
    centroid_poses = []
    fused_centroid_path = []
    tx_rel = []
    ty_rel = []
    tz_rel = []

    total_frames = max(p[0] for p in all_frame_poses) + 1  # Assuming frame indices start from 0
    if (end_frame is None):
        end_frame = total_frames
    
    print("Total Frames in Data: ", total_frames)
    print(f"Plotting Centroid Path from frame {start_frame} to {end_frame}")

    for frame_idx in range(start_frame, end_frame):
        # Collecting same frame index poses, all tags.
        current_frame_poses = get_current_frame_poses(all_frame_poses, frame_idx)

        
        # Iterate through all tag paths and plot their centroid paths
        for pose in current_frame_poses:
            centroids = []
            # for pose in poses:
            tag_id = pose[0]
            _, _, tx, ty, tz, qx, qy, qz, qw = pose
            t = np.array([[1, 0, 0, tx],
                            [0, 1, 0, ty],
                            [0, 0, 1, tz],
                            [0, 0, 0, 1]])
            if tag_id == 1:
                T_centroid = T_C1 @ t
            elif tag_id == 2:
                T_centroid = T_C2 @ t
            elif tag_id == 3:
                T_centroid = T_C3 @ t
            elif tag_id == 4:
                T_centroid = T_C4 @ t
            elif tag_id == 5:
                T_centroid = T_C5 @ t
            elif tag_id == 0:
                static_tag_pose = pose            
            else:
                T_centroid = np.eye(4)

            if (tag_id != 0):
                # Extract translation and orientation
                tx, ty, tz = T_centroid[:3, 3]
                rotation_matrix = T_centroid[:3, :3]
                quaternion = R.from_matrix(rotation_matrix).as_quat()  # (qx, qy, qz, qw)
                qx, qy, qz, qw = quaternion
                centroid = [tx, ty, tz, qx, qy, qz, qw]
                centroid_poses.append(centroid)

        # Fuse the centroid poses from all visible tags to get a more stable estimate.
        fused_pose = pose_fusion(centroid_poses)

        # TODO: No visible tags case: currently returning None, should handle it better.
        if fused_pose is not None:
            relative_fused_pose = []

            # Relative to static tag logic to be added here.
            # static_tag_pose --> (frame_idx, tx, ty, tz, qx, qy, qz, qw)
            # fused_pose --> (tx, ty, tz, qx, qy, qz, qw)
            # translation relative to static tag
            relative_fused_pose.append(fused_pose[0] - static_tag_pose[1])
            relative_fused_pose.append(fused_pose[1] - static_tag_pose[2])
            relative_fused_pose.append(fused_pose[2] - static_tag_pose[3])
            # rotation relative to static tag
            R_fused = R.from_quat(fused_pose[3:7]).as_matrix()
            R_static = R.from_quat(static_tag_pose[5:9]).as_matrix()
            R_rel = R_static.T @ R_fused
            quaternion_rel = R.from_matrix(R_rel).as_quat()
            relative_fused_pose.append(quaternion_rel[0])
            relative_fused_pose.append(quaternion_rel[1])
            relative_fused_pose.append(quaternion_rel[2])
            relative_fused_pose.append(quaternion_rel[3])

            fused_centroid_path.append(relative_fused_pose)




    # if fused_pose is None: 
    # TODO: change the logic, to have empty or something printed out not just return. no visible tags
    #     return
    # fused_centroid_path = np.array(fused_centroid_path)

    if len(fused_centroid_path) > 0:
        ax_centroid.plot(np.array(fused_centroid_path)[:, 0], np.array(fused_centroid_path)[:, 1], np.array(fused_centroid_path)[:, 2], c='r')
    else:
        pass # or use fused_centroid_path = [[0,0,0]] to plot a point at origin.

    # Set labels and title
    # I want all x,y and z range to be same for better visualization: ans use the largest range among them.
    
    ax_centroid.set_xlabel('X (m)')
    ax_centroid.set_ylabel('Y (m)')
    ax_centroid.set_zlabel('Z (m)')

    if axis_equal:
        ax_centroid.set_box_aspect([1,1,1])  # Aspect ratio is 1:1:1

        # --- Make equal scale automatically ---
        x = np.array(fused_centroid_path)[:, 0]
        y = np.array(fused_centroid_path)[:, 1]
        z = np.array(fused_centroid_path)[:, 2]
        max_range = np.ptp([x, y, z]).max() / 2.0
        mid_x = (np.max(x) + np.min(x)) / 2.0
        mid_y = (np.max(y) + np.min(y)) / 2.0
        mid_z = (np.max(z) + np.min(z)) / 2.0

        ax_centroid.set_xlim(mid_x - max_range, mid_x + max_range)
        ax_centroid.set_ylim(mid_y - max_range, mid_y + max_range)
        ax_centroid.set_zlim(mid_z - max_range, mid_z + max_range)



    ax_centroid.set_title('Cube Centroid Path')
    ax_centroid.legend()
    plt.show()

'''
Plot all external tag paths in 3D space.
'''
def plot_all_ext_tags(ax_all, all_frame_poses):
    ax_all.clear()  # Clear the previous plot
    tag_paths = {}
    for row in all_frame_poses:
        tag_id = int(row[0]) #tag_id
        pose = (
            int(row[1]), #frame_id
            float(row[2]),
            float(row[3]),
            float(row[4]),
            float(row[5]),
            float(row[6]),
            float(row[7]),
            float(row[8])
        )
        tag_paths.setdefault(tag_id, []).append(pose)
    # Plot each tag's path
    for tag_id, poses in tag_paths.items():
        poses = sorted(poses, key=lambda x: x[0])
        tx = [p[1] for p in poses]
        ty = [p[2] for p in poses]
        tz = [p[3] for p in poses]
        ax_all.plot(tx, ty, tz, label=f'Tag {tag_id}')
    ax_all.set_xlabel('X (m)')
    ax_all.set_ylabel('Y (m)')
    ax_all.set_zlabel('Z (m)')
    ax_all.legend()
    ax_all.set_title('All External Tag Paths')
    plt.show()

#######################################################################################
#          Plot the Relative Tag Pose for all Internal Tags wrt Timestamp
#######################################################################################



#######################################################################################
#                           Telemetry Data Plot
#######################################################################################



#######################################################################################
#                           Execution Loop
#######################################################################################

if __name__ == "__main__":


    ###########################
            # Internal
    ###########################

    # fig = plt.figure(figsize=(12, 6))
    # ax3 = fig.add_subplot(111, projection='3d')

    # ax4 = fig.add_subplot(234, projection='3d')
    # ax5 = fig.add_subplot(235, projection='3d')

    # Create subplots for x, y, z signals: for plot indv function.
    fig, (ax_x, ax_y, ax_z) = plt.subplots(3, 1, figsize=(8, 12))
    fig.suptitle(f'Separate Axes of Tag Over Time')

    ###########################    
            # External
    ###########################

    # # Creating the figure and axes once for cube centroid plotting.
    # fig_centroid = plt.figure()
    # ax_centroid = fig_centroid.add_subplot(111, projection='3d')

    ###########################
            # Telemetry
    ###########################


    while True:

        ###########################
                # Internal
        ###########################
        
        # rel_tag_poses = filter_false_positive()
        rel_tag_poses = unfilter_rel_paths()
        # plot_filtered_tags(rel_tag_poses, ax_x, ax_y, ax_z, tagA=5, tagB=0, tagC=14, tagD=5)
        plot_filtered_tags_displacement(rel_tag_poses, ax_x, ax_y, ax_z, tagA=5, tagD=1)
        
        


        # plot_idv(17, ax_x, ax_y, ax_z)                 # IMP ## x,y,z signals wrt time for a tag.


        # Unused: plot_idv_denoised( 19, 1, ax5, ax_x, ax_y, ax_z)  ## x,y,z signals wrt time for a tag denoised by static tag
        # plot_6d_pose(ax1)
        # plot_3d_relative_pose(ax2)
        # plot_relative_pose_indv(1, ax3) 

        ###########################
                # External
        ###########################

        # # Read from csv to get the data in this form:
        # all_frame_poses = read_frame_poses()
        # curr_tag_id = 5
        # # plot_all_ext_tags(ax_centroid, all_frame_poses)
        # plot_centroid_path(ax_centroid, all_frame_poses, axis_equal=False)


        ###########################
                # Telemetry
        ###########################




        
















# TODO: 
# 1. Add checks for stability values correponsiding to both int and ext and use frames only when both are stable.
# I am doing per frame classification and not memory buffer based regression or prediction, so this should be fine.
# 2. denoising/ comparision funtion for internal tags does not make a lot of sense right now.

# To Note IMP:  
# before usinf the ratation part for the static tag relaative transform, and after the plot remains the same : Wierd, considering 45 degree tilt not same plane...??
