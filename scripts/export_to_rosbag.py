#!/usr/bin/env python3
"""
Export point cloud data to ROS bag file for RViz visualization.

This script reads the output from demo.py and converts it to ROS bag format.
It publishes point clouds and camera poses that can be visualized in RViz.

Usage:
    python export_to_rosbag.py --output_dir OUTPUT_DIR --bag_file BAG_FILE [--frame_rate FRAME_RATE] [--downsample DOWNSAMPLE]

Example:
    python export_to_rosbag.py --output_dir demo_tmp --bag_file pointcloud.bag --frame_rate 30 --downsample 100
"""

import os
import sys
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R

# Try to import ROS modules
try:
    import rospy
    import rosbag
    from sensor_msgs.msg import PointCloud2, PointField
    from geometry_msgs.msg import PoseStamped, Point, Vector3, TransformStamped, TwistWithCovariance, PoseWithCovariance
    from nav_msgs.msg import Odometry, Path
    from tf2_msgs.msg import TFMessage
    from visualization_msgs.msg import Marker
    from std_msgs.msg import Header, ColorRGBA
    ROS_AVAILABLE = True
except ImportError as e:
    ROS_AVAILABLE = False
    print("Warning: ROS modules not available. Please install ROS or rosbag package.")
    print(f"Error: {e}")
    print("\nTo install ROS dependencies:")
    print("  For ROS Noetic: sudo apt-get install ros-noetic-rosbag")
    print("  Or install Python packages: pip install rospkg catkin_pkg")
    sys.exit(1)


def numpy_to_pointcloud2(points, colors, confidence=None, frame_id="map", timestamp=None):
    """
    Convert numpy arrays to ROS PointCloud2 message.
    
    Args:
        points: numpy array of shape [N, 3] containing xyz coordinates
        colors: numpy array of shape [N, 3] containing RGB colors (0-1 range)
        confidence: numpy array of shape [N] containing confidence values (optional)
        frame_id: ROS frame ID
        timestamp: rospy.Time object, if None uses current time
    
    Returns:
        sensor_msgs.PointCloud2 message
    """
    if timestamp is None:
        timestamp = rospy.Time.now()
    
    # Ensure points and colors are contiguous arrays
    points = np.ascontiguousarray(points, dtype=np.float32)
    colors = np.ascontiguousarray(colors, dtype=np.float32)
    
    # Clamp colors to [0, 1] and convert to uint8
    colors_uint8 = (np.clip(colors, 0, 1) * 255).astype(np.uint8)
    
    # Pack RGB into UINT32 format: R << 16 | G << 8 | B
    # Note: RViz expects RGB in this format
    rgb_packed = (colors_uint8[:, 0].astype(np.uint32) << 16) | \
                 (colors_uint8[:, 1].astype(np.uint32) << 8) | \
                 colors_uint8[:, 2].astype(np.uint32)
    
    # Combine points, colors, and confidence
    N = points.shape[0]
    
    if confidence is not None:
        confidence = np.ascontiguousarray(confidence, dtype=np.float32)
        dtype_list = [
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('rgb', np.uint32),
            ('confidence', np.float32),
        ]
    else:
        dtype_list = [
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('rgb', np.uint32),
        ]
    
    # Create structured array
    cloud_array = np.empty(N, dtype=dtype_list)
    cloud_array['x'] = points[:, 0]
    cloud_array['y'] = points[:, 1]
    cloud_array['z'] = points[:, 2]
    cloud_array['rgb'] = rgb_packed
    if confidence is not None:
        cloud_array['confidence'] = confidence
    
    # Create PointCloud2 message
    msg = PointCloud2()
    msg.header = Header()
    msg.header.stamp = timestamp
    msg.header.frame_id = frame_id
    
    msg.height = 1
    msg.width = N
    
    if confidence is not None:
        msg.fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('rgb', 12, PointField.UINT32, 1),
            PointField('confidence', 16, PointField.FLOAT32, 1),
        ]
        msg.point_step = 20  # 3*4 (float32) + 1*4 (uint32 rgb) + 1*4 (confidence float32) = 20 bytes
    else:
        msg.fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('rgb', 12, PointField.UINT32, 1),
        ]
        msg.point_step = 16  # 3*4 (float32) + 1*4 (uint32 rgb) = 16 bytes
    
    msg.is_bigendian = False
    msg.row_step = msg.point_step * N
    msg.is_dense = True
    msg.data = cloud_array.tobytes()
    
    return msg


def matrix_to_pose_stamped(c2w_matrix, frame_id="map", timestamp=None):
    """
    Convert 4x4 camera-to-world matrix to ROS PoseStamped message.
    
    Args:
        c2w_matrix: 4x4 numpy array representing camera-to-world transformation
        frame_id: ROS frame ID
        timestamp: rospy.Time object, if None uses current time
    
    Returns:
        geometry_msgs.PoseStamped message
    """
    if timestamp is None:
        timestamp = rospy.Time.now()
    
    # Extract rotation and translation
    R_matrix = c2w_matrix[:3, :3]
    t = c2w_matrix[:3, 3]
    
    # Convert rotation matrix to quaternion
    rotation = R.from_matrix(R_matrix)
    quat = rotation.as_quat()  # [x, y, z, w]
    
    # Create PoseStamped message
    pose = PoseStamped()
    pose.header = Header()
    pose.header.stamp = timestamp
    pose.header.frame_id = frame_id
    
    pose.pose.position.x = t[0]
    pose.pose.position.y = t[1]
    pose.pose.position.z = t[2]
    
    pose.pose.orientation.x = quat[0]
    pose.pose.orientation.y = quat[1]
    pose.pose.orientation.z = quat[2]
    pose.pose.orientation.w = quat[3]
    
    return pose


def matrix_to_odometry(c2w_matrix, frame_id="map", child_frame_id="base_link", timestamp=None, 
                       prev_pose=None, prev_timestamp=None, linear_velocity=None, angular_velocity=None):
    """
    Convert 4x4 camera-to-world matrix to ROS Odometry message.
    
    Args:
        c2w_matrix: 4x4 numpy array representing camera-to-world transformation
        frame_id: Parent frame ID (default: "map")
        child_frame_id: Child frame ID (default: "base_link")
        timestamp: rospy.Time object, if None uses current time
        prev_pose: Previous pose matrix (4x4) for velocity calculation (optional)
        prev_timestamp: Previous timestamp for velocity calculation (optional)
        linear_velocity: Linear velocity [vx, vy, vz] in m/s (optional, will be calculated if prev_pose provided)
        angular_velocity: Angular velocity [wx, wy, wz] in rad/s (optional, will be calculated if prev_pose provided)
    
    Returns:
        nav_msgs.Odometry message
    """
    if timestamp is None:
        timestamp = rospy.Time.now()
    
    # Extract rotation and translation
    R_matrix = c2w_matrix[:3, :3]
    t = c2w_matrix[:3, 3]
    
    # Convert rotation matrix to quaternion
    rotation = R.from_matrix(R_matrix)
    quat = rotation.as_quat()  # [x, y, z, w]
    
    # Create Odometry message
    odom = Odometry()
    odom.header = Header()
    odom.header.stamp = timestamp
    odom.header.frame_id = frame_id
    odom.child_frame_id = child_frame_id
    
    # Set pose
    odom.pose.pose.position.x = t[0]
    odom.pose.pose.position.y = t[1]
    odom.pose.pose.position.z = t[2]
    odom.pose.pose.orientation.x = quat[0]
    odom.pose.pose.orientation.y = quat[1]
    odom.pose.pose.orientation.z = quat[2]
    odom.pose.pose.orientation.w = quat[3]
    
    # Set pose covariance (6x6 matrix, row-major order)
    # Default: small uncertainty for position, larger for orientation
    pose_cov = np.zeros(36)
    pose_cov[0] = 0.01   # x
    pose_cov[7] = 0.01   # y
    pose_cov[14] = 0.01  # z
    pose_cov[21] = 0.1   # roll
    pose_cov[28] = 0.1   # pitch
    pose_cov[35] = 0.1   # yaw
    odom.pose.covariance = pose_cov.tolist()
    
    # Calculate or set velocity
    if linear_velocity is not None:
        odom.twist.twist.linear.x = linear_velocity[0]
        odom.twist.twist.linear.y = linear_velocity[1]
        odom.twist.twist.linear.z = linear_velocity[2]
        if angular_velocity is not None:
            odom.twist.twist.angular.x = angular_velocity[0]
            odom.twist.twist.angular.y = angular_velocity[1]
            odom.twist.twist.angular.z = angular_velocity[2]
        else:
            odom.twist.twist.angular.x = 0.0
            odom.twist.twist.angular.y = 0.0
            odom.twist.twist.angular.z = 0.0
    elif prev_pose is not None and prev_timestamp is not None:
        # Calculate velocity from previous pose
        dt = (timestamp - prev_timestamp).to_sec()
        if dt > 0:
            prev_t = prev_pose[:3, 3]
            # Linear velocity in world frame
            linear_vel = (t - prev_t) / dt
            odom.twist.twist.linear.x = linear_vel[0]
            odom.twist.twist.linear.y = linear_vel[1]
            odom.twist.twist.linear.z = linear_vel[2]
            
            # Angular velocity
            prev_R = prev_pose[:3, :3]
            prev_rot = R.from_matrix(prev_R)
            # Relative rotation
            rel_rot = rotation * prev_rot.inv()
            # Convert to axis-angle and divide by dt
            axis_angle = rel_rot.as_rotvec()
            angular_vel = axis_angle / dt
            odom.twist.twist.angular.x = angular_vel[0]
            odom.twist.twist.angular.y = angular_vel[1]
            odom.twist.twist.angular.z = angular_vel[2]
        else:
            # Zero velocity if dt is zero or negative
            odom.twist.twist.linear.x = 0.0
            odom.twist.twist.linear.y = 0.0
            odom.twist.twist.linear.z = 0.0
            odom.twist.twist.angular.x = 0.0
            odom.twist.twist.angular.y = 0.0
            odom.twist.twist.angular.z = 0.0
    else:
        # Zero velocity if not provided
        odom.twist.twist.linear.x = 0.0
        odom.twist.twist.linear.y = 0.0
        odom.twist.twist.linear.z = 0.0
        odom.twist.twist.angular.x = 0.0
        odom.twist.twist.angular.y = 0.0
        odom.twist.twist.angular.z = 0.0
    
    # Set twist covariance (6x6 matrix, row-major order)
    twist_cov = np.zeros(36)
    twist_cov[0] = 0.1   # linear x
    twist_cov[7] = 0.1   # linear y
    twist_cov[14] = 0.1  # linear z
    twist_cov[21] = 0.1  # angular x
    twist_cov[28] = 0.1  # angular y
    twist_cov[35] = 0.1  # angular z
    odom.twist.covariance = twist_cov.tolist()
    
    return odom


def create_path_from_poses(pose_stamped_list, frame_id="map", timestamp=None):
    """
    Create a Path message from a list of PoseStamped messages.
    
    Args:
        pose_stamped_list: List of geometry_msgs.PoseStamped messages
        frame_id: Frame ID for the path (default: "map")
        timestamp: rospy.Time object for the path header, if None uses current time
    
    Returns:
        nav_msgs.Path message
    """
    if timestamp is None:
        timestamp = rospy.Time.now()
    
    path = Path()
    path.header = Header()
    path.header.stamp = timestamp
    path.header.frame_id = frame_id
    path.poses = pose_stamped_list
    
    return path


def matrix_to_tf_transform(c2w_matrix, child_frame_id, parent_frame_id="map", timestamp=None):
    """
    Convert 4x4 camera-to-world matrix to ROS TransformStamped message for TF.
    This allows RViz Axes display to show coordinate frames.
    
    Args:
        c2w_matrix: 4x4 numpy array representing camera-to-world transformation
        child_frame_id: Child frame ID (e.g., "camera_0")
        parent_frame_id: Parent frame ID (default: "map")
        timestamp: rospy.Time object, if None uses current time
    
    Returns:
        geometry_msgs.TransformStamped message
    """
    if timestamp is None:
        timestamp = rospy.Time.now()
    
    # Extract rotation and translation
    R_matrix = c2w_matrix[:3, :3]
    t = c2w_matrix[:3, 3]
    
    # Convert rotation matrix to quaternion
    rotation = R.from_matrix(R_matrix)
    quat = rotation.as_quat()  # [x, y, z, w]
    
    # Create TransformStamped message
    transform = TransformStamped()
    transform.header = Header()
    transform.header.stamp = timestamp
    transform.header.frame_id = parent_frame_id
    transform.child_frame_id = child_frame_id
    
    transform.transform.translation.x = t[0]
    transform.transform.translation.y = t[1]
    transform.transform.translation.z = t[2]
    
    transform.transform.rotation.x = quat[0]
    transform.transform.rotation.y = quat[1]
    transform.transform.rotation.z = quat[2]
    transform.transform.rotation.w = quat[3]
    
    return transform


def matrix_to_axis_marker(c2w_matrix, frame_id="map", timestamp=None, scale=0.1, marker_id=0):
    """
    Convert 4x4 camera-to-world matrix to ROS Marker message for axis visualization.
    Creates three lines (red=X, green=Y, blue=Z) representing the coordinate frame.
    
    Args:
        c2w_matrix: 4x4 numpy array representing camera-to-world transformation
        frame_id: ROS frame ID
        timestamp: rospy.Time object, if None uses current time
        scale: Length of each axis line (default: 0.1)
        marker_id: Unique ID for the marker (default: 0)
    
    Returns:
        visualization_msgs.Marker message (LINE_LIST type)
    """
    if timestamp is None:
        timestamp = rospy.Time.now()
    
    # Extract rotation and translation
    R_matrix = c2w_matrix[:3, :3]
    t = c2w_matrix[:3, 3]
    
    # Create axis vectors in camera frame (normalized)
    axis_length = scale
    x_axis = np.array([axis_length, 0, 0])
    y_axis = np.array([0, axis_length, 0])
    z_axis = np.array([0, 0, axis_length])
    
    # Transform to world frame
    x_axis_world = (R_matrix @ x_axis) + t
    y_axis_world = (R_matrix @ y_axis) + t
    z_axis_world = (R_matrix @ z_axis) + t
    
    # Create Marker message
    marker = Marker()
    marker.header = Header()
    marker.header.stamp = timestamp
    marker.header.frame_id = frame_id
    
    marker.ns = "camera_axes"
    marker.id = marker_id
    marker.type = Marker.LINE_LIST  # Use LINE_LIST instead of ARROW_LIST
    marker.action = Marker.ADD
    
    # Set scale (line width)
    marker.scale.x = scale * 0.05  # line width
    
    # Set color and points for each axis
    # X axis (red)
    marker.points.append(Point(x=t[0], y=t[1], z=t[2]))
    marker.points.append(Point(x=x_axis_world[0], y=x_axis_world[1], z=x_axis_world[2]))
    marker.colors.append(ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0))
    marker.colors.append(ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0))
    
    # Y axis (green)
    marker.points.append(Point(x=t[0], y=t[1], z=t[2]))
    marker.points.append(Point(x=y_axis_world[0], y=y_axis_world[1], z=y_axis_world[2]))
    marker.colors.append(ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0))
    marker.colors.append(ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0))
    
    # Z axis (blue)
    marker.points.append(Point(x=t[0], y=t[1], z=t[2]))
    marker.points.append(Point(x=z_axis_world[0], y=z_axis_world[1], z=z_axis_world[2]))
    marker.colors.append(ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0))
    marker.colors.append(ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0))
    
    marker.lifetime = rospy.Duration(0)  # 0 means never auto-delete
    marker.frame_locked = False
    
    return marker


def load_data_from_output_dir(output_dir, downsample=100):
    """
    Load point cloud data from output directory created by demo.py.
    
    Args:
        output_dir: Directory containing depth/, color/, conf/, camera/ subdirectories
        downsample: Downsampling factor - keep 1 point every N points (default: 100)
    
    Returns:
        tuple: (points_list, colors_list, confidence_list, poses_list)
    """
    depth_dir = os.path.join(output_dir, "depth")
    color_dir = os.path.join(output_dir, "color")
    conf_dir = os.path.join(output_dir, "conf")
    camera_dir = os.path.join(output_dir, "camera")
    
    if not os.path.exists(depth_dir) or not os.path.exists(color_dir) or not os.path.exists(camera_dir):
        raise ValueError(f"Output directory {output_dir} does not contain required subdirectories")
    
    # Check if confidence directory exists
    has_confidence = os.path.exists(conf_dir)
    if not has_confidence:
        print(f"Warning: No confidence directory found in {output_dir}")
    
    # Get all frame files
    depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith('.npy')])
    color_files = sorted([f for f in os.listdir(color_dir) if f.endswith('.png')])
    camera_files = sorted([f for f in os.listdir(camera_dir) if f.endswith('.npz')])
    
    if has_confidence:
        conf_files = sorted([f for f in os.listdir(conf_dir) if f.endswith('.npy')])
        if len(depth_files) != len(conf_files):
            print(f"Warning: Mismatch in number of confidence files. Depth: {len(depth_files)}, Conf: {len(conf_files)}")
    
    if len(depth_files) != len(color_files) or len(depth_files) != len(camera_files):
        print(f"Warning: Mismatch in number of files. Depth: {len(depth_files)}, Color: {len(color_files)}, Camera: {len(camera_files)}")
    
    num_frames = min(len(depth_files), len(color_files), len(camera_files))
    
    points_list = []
    colors_list = []
    confidence_list = []
    poses_list = []
    
    print(f"Loading {num_frames} frames from {output_dir}...")
    
    for i in range(num_frames):
        # Load depth
        depth_path = os.path.join(depth_dir, depth_files[i])
        depth = np.load(depth_path)
        
        # Load color
        import cv2
        color_path = os.path.join(color_dir, color_files[i])
        color_bgr = cv2.imread(color_path)
        color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB) / 255.0
        
        # Load confidence if available
        if has_confidence and i < len(conf_files):
            conf_path = os.path.join(conf_dir, conf_files[i])
            confidence = np.load(conf_path)
        else:
            confidence = None
        
        # Load camera pose
        camera_path = os.path.join(camera_dir, camera_files[i])
        camera_data = np.load(camera_path)
        c2w = camera_data['pose']  # 4x4 matrix
        
        # Use the points directly from the saved data if available
        # Otherwise, reconstruct from depth
        # For now, we'll use the depth to reconstruct points
        # Handle different depth array shapes (2D, 3D with channel dimension, etc.)
        if depth.ndim > 2:
            # If depth has more than 2 dimensions, squeeze out singleton dimensions
            depth = np.squeeze(depth)
        if depth.ndim != 2:
            raise ValueError(f"Expected depth to be 2D after squeezing, but got shape {depth.shape}")
        H, W = depth.shape
        
        # Handle confidence array shape to match depth
        if confidence is not None:
            # Handle different confidence array shapes
            if confidence.ndim > 2:
                confidence = np.squeeze(confidence)
            # Ensure confidence matches depth dimensions
            if confidence.ndim == 2 and confidence.shape != (H, W):
                # If shapes don't match after squeezing, try to reshape
                if confidence.size == H * W:
                    confidence = confidence.reshape(H, W)
                else:
                    print(f"Warning: Confidence shape {confidence.shape} doesn't match depth shape ({H}, {W})")
        intrinsics = camera_data['intrinsics']
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        
        # Generate pixel coordinates
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        
        # Convert to 3D points in camera frame
        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth
        
        # Stack into point cloud (camera frame)
        points_cam = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        
        # Transform to world coordinates using c2w matrix
        points_homogeneous = np.hstack([points_cam, np.ones((points_cam.shape[0], 1))])
        points_world = (c2w @ points_homogeneous.T).T[:, :3]
        
        # Reshape colors to match points
        colors = color_rgb.reshape(-1, 3)
        
        # Reshape confidence to match points if available
        if confidence is not None:
            confidence_flat = confidence.reshape(-1)
        else:
            confidence_flat = None
        
        # Filter out invalid points (zero depth or NaN)
        valid_mask = (depth.reshape(-1) > 0) & ~np.isnan(points_world).any(axis=1)
        points_world = points_world[valid_mask]
        colors = colors[valid_mask]
        if confidence_flat is not None:
            confidence_flat = confidence_flat[valid_mask]
        
        # Downsample point cloud
        if downsample > 1 and len(points_world) > 0:
            indices = np.arange(0, len(points_world), downsample)
            points_world = points_world[indices]
            colors = colors[indices]
            if confidence_flat is not None:
                confidence_flat = confidence_flat[indices]
        
        points_list.append(points_world)
        colors_list.append(colors)
        confidence_list.append(confidence_flat)
        poses_list.append(c2w)
        
        if (i + 1) % 10 == 0:
            print(f"  Loaded {i + 1}/{num_frames} frames...")
    
    print(f"Successfully loaded {num_frames} frames")
    return points_list, colors_list, confidence_list, poses_list


def export_to_rosbag(output_dir, bag_file, frame_rate=30.0, downsample=100):
    """
    Export point cloud data to ROS bag file.
    
    Args:
        output_dir: Directory containing the output data from demo.py
        bag_file: Output ROS bag file path
        frame_rate: Frame rate for the bag file (Hz)
        downsample: Downsampling factor - keep 1 point every N points (default: 100)
    """
    # Load data
    points_list, colors_list, confidence_list, poses_list = load_data_from_output_dir(output_dir, downsample=downsample)
    
    num_frames = len(points_list)
    frame_duration = rospy.Duration(1.0 / frame_rate)
    
    has_confidence = confidence_list[0] is not None if confidence_list else False
    
    print(f"\nWriting ROS bag file: {bag_file}")
    print(f"  Frames: {num_frames}")
    print(f"  Frame rate: {frame_rate} Hz")
    print(f"  Duration: {num_frames / frame_rate:.2f} seconds")
    print(f"  Downsampling: {downsample}x (keep 1 point every {downsample} points)")
    print(f"  Confidence data: {'Yes' if has_confidence else 'No'}")
    print(f"  Topics:")
    print(f"    - /pointcloud")
    print(f"    - /camera_pose")
    print(f"    - /odom")
    print(f"    - /path")
    print(f"    - /tf (for RViz Axes display)")
    
    # Initialize ROS (required for creating messages)
    rospy.init_node('ttt3r_rosbag_exporter', anonymous=True)
    
    # Open bag file for writing
    bag = rosbag.Bag(bag_file, 'w')
    
    try:
        start_time = rospy.Time.now()
        prev_pose = None
        prev_timestamp = None
        path_poses = []  # Accumulate poses for Path message
        
        for i in range(num_frames):
            timestamp = start_time + frame_duration * i
            
            # Write point cloud with confidence if available
            pc_msg = numpy_to_pointcloud2(
                points_list[i],
                colors_list[i],
                confidence=confidence_list[i],
                frame_id="map",
                timestamp=timestamp
            )
            bag.write("/pointcloud", pc_msg, timestamp)
            
            # Write camera pose
            pose_msg = matrix_to_pose_stamped(
                poses_list[i],
                frame_id="map",
                timestamp=timestamp
            )
            bag.write("/camera_pose", pose_msg, timestamp)
            
            # Accumulate pose for Path
            path_poses.append(pose_msg)
            
            # Write Path (updated with all poses up to current frame)
            path_msg = create_path_from_poses(
                path_poses,
                frame_id="map",
                timestamp=timestamp
            )
            bag.write("/path", path_msg, timestamp)
            
            # Write odometry (with velocity calculated from previous pose if available)
            odom_msg = matrix_to_odometry(
                poses_list[i],
                frame_id="map",
                child_frame_id="camera",
                timestamp=timestamp,
                prev_pose=prev_pose,
                prev_timestamp=prev_timestamp
            )
            bag.write("/odom", odom_msg, timestamp)
            
            # Write TF transform for RViz Axes display
            # Use a fixed frame ID "camera" so axes will follow the current pose
            tf_transform = matrix_to_tf_transform(
                poses_list[i],
                child_frame_id="camera",
                parent_frame_id="map",
                timestamp=timestamp
            )
            tf_msg = TFMessage()
            tf_msg.transforms = [tf_transform]
            bag.write("/tf", tf_msg, timestamp)
            
            # Update previous pose and timestamp for next iteration
            prev_pose = poses_list[i]
            prev_timestamp = timestamp
            
            if (i + 1) % 10 == 0:
                print(f"  Written {i + 1}/{num_frames} frames...")
        
        print(f"\nSuccessfully wrote {num_frames} frames to {bag_file}")
        print(f"Bag file size: {os.path.getsize(bag_file) / (1024*1024):.2f} MB")
        
    finally:
        bag.close()
        print("Bag file closed.")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Export TTT3R output to ROS bag file for RViz visualization."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory containing output data from demo.py (should contain depth/, color/, camera/ subdirectories)",
    )
    parser.add_argument(
        "--bag_file",
        type=str,
        required=True,
        help="Output ROS bag file path (e.g., pointcloud.bag)",
    )
    parser.add_argument(
        "--frame_rate",
        type=float,
        default=30.0,
        help="Frame rate for the bag file in Hz (default: 30.0)",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=100,
        help="Downsampling factor - keep 1 point every N points (default: 100)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    if not os.path.exists(args.output_dir):
        print(f"Error: Output directory does not exist: {args.output_dir}")
        sys.exit(1)
    
    # Create output directory for bag file if needed
    bag_dir = os.path.dirname(args.bag_file)
    if bag_dir and not os.path.exists(bag_dir):
        os.makedirs(bag_dir, exist_ok=True)
    
    try:
        export_to_rosbag(
            args.output_dir,
            args.bag_file,
            args.frame_rate,
            args.downsample
        )
        print("\nExport completed successfully!")
        print(f"\nTo play the bag file in RViz:")
        print(f"  rosbag play {args.bag_file}")
        print(f"\nTopics in the bag:")
        print(f"  - /pointcloud (sensor_msgs/PointCloud2)")
        print(f"  - /camera_pose (geometry_msgs/PoseStamped)")
        print(f"  - /odom (nav_msgs/Odometry) - with pose and velocity")
        print(f"  - /path (nav_msgs/Path) - camera trajectory path")
        print(f"  - /tf (tf2_msgs/TFMessage) - for RViz Axes display")
        print(f"\nTo view camera axes in RViz:")
        print(f"  1. Add 'Axes' display")
        print(f"  2. Set Reference Frame to 'camera' (or 'map' to see all frames)")
        print(f"  3. The axes will follow the camera pose as the bag plays")
        print(f"  4. Alternatively, set Reference Frame to 'map' and Length to see axes at each pose")
        print(f"\nTo view camera path in RViz:")
        print(f"  1. Add 'Path' display")
        print(f"  2. Set Topic to '/path'")
        print(f"  3. Set Fixed Frame to 'map'")
        print(f"  4. The path will show the camera trajectory as the bag plays")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

