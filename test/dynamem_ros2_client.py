#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist, PoseStamped
from std_msgs.msg import Float32MultiArray
# from sensor_msgs import point_cloud2
from sensor_msgs_py import point_cloud2
from cv_bridge import CvBridge
import cv2
import math
from threading import Lock
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped

from typing import Dict, List, Optional
from stretch.core import AbstractRobotClient
from stretch.motion import Footprint, RobotModel
# from stretch.motion.kinematics import HelloStretchKinematics
from stretch.utils.geometry import xyt_base_to_global
from stretch.visualization.rerun import RerunVisualizer



class DynaMemROS2Client(AbstractRobotClient, Node):
    """ROS2 client for DynaMem that interfaces directly with simulation topics."""
    
    def __init__(self, node_name='dynamem_ros2_client'):
        # Initialize ROS2 if not already initialized
        if not rclpy.ok():
            rclpy.init()
            
        # Initialize AbstractRobotClient with no arguments
        AbstractRobotClient.__init__(self)
        
        # Initialize Node with the node name
        Node.__init__(self, node_name)
        
        # Initialize state variables
        self._obs_lock = Lock()
        self._state_lock = Lock()
        self._current_rgb = None
        self._current_depth = None
        self._current_xyz = None
        self._current_camera_info = None
        self._current_pose = None
        self._current_joints = None
        self._at_goal = False
        self._control_mode = "navigation"
        
        # Initialize ROS2 components
        self.bridge = CvBridge()
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # Publishers
        self._cmd_vel_pub = self.create_publisher(Twist, '/stretch/cmd_vel', 10)
        self._joint_cmd_pub = self.create_publisher(Float32MultiArray, '/joint_pose_cmd', 10)
        self._goal_pose_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        
        # Subscribers
        self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self._color_image_callback,
            10
        )
        
        self.create_subscription(
            Image,
            '/camera/depth/image_rect_raw',
            self._depth_image_callback,
            10
        )
        
        self.create_subscription(
            CameraInfo,
            '/camera/color/camera_info',
            self._camera_info_callback,
            10
        )
        
        self.create_subscription(
            PointCloud2,
            '/camera/depth/color/points',
            self._point_cloud_callback,
            10
        )
        
        self.create_subscription(
            Odometry,
            '/odom',
            self._odom_callback,
            10
        )
        
        self.create_subscription(
            Float32MultiArray,
            '/joint_states',
            self._joint_states_callback,
            10
        )

        self._robot_model = self
        self.dof = 3 + 2 + 4 + 2
        self.xyt = np.zeros(3)

        enable_rerun_server = True
        output_path = None
        # Optional live visualization
        if enable_rerun_server:
            self._rerun = RerunVisualizer(output_path=output_path)
        else:
            self._rerun = None
            self._rerun_thread = None
        
        self.get_logger().info("DynaMem ROS2 client initialized")


    # --- Needed ---
    def execute_trajectory(self, trajectory, pos_err_threshold=0.2, rot_err_threshold=0.1,
                         spin_rate=10, verbose=False, per_waypoint_timeout=10.0,
                         final_timeout=10.0, relative=False, blocking=False):
        """Execute a sequence of base waypoints."""
        for waypoint in trajectory:
            self.move_base_to(waypoint, relative=relative, blocking=blocking)
        return True

    def get_pose_graph(self) -> np.ndarray:
        """Get SLAM pose graph (simplified for simulation)."""
        # Return a dummy pose graph for simulation
        return np.array([])

    def move_to_manip_posture(self) -> None:
        """Move to manipulation posture (simplified for simulation)."""
        # Example joint positions for manipulation posture
        manip_posture = np.array([0.0, 0.5, 0.0, 0.0, 0.0, 0.0])  # Adjust as needed
        self.robot_to(manip_posture, blocking=True)

    def switch_to_navigation_mode(self):
        # return self._base_control_mode == ControlMode.NAVIGATION
        return True

    def get_robot_model(self) -> RobotModel:
        """return a model of the robot for planning"""
        return self._robot_model
    
    def start(self):
        return True

    # --- Sensor Getters ---
    def get_ee_rgbd(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get end-effector RGB and depth images."""
        with self._obs_lock:
            # In simulation, we'll use the main camera for both head and ee
            return self._current_rgb, self._current_depth

    def get_head_rgbd(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get head RGB and depth images."""
        with self._obs_lock:
            return self._current_rgb, self._current_depth

    # --- State Getters ---
    def get_joint_state(self, timeout: float = 5.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get joint positions, velocities, and efforts."""
        with self._state_lock:
            if self._current_joints is None:
                return None, None, None
            # In simulation, we may not have velocities/efforts
            return self._current_joints, np.zeros_like(self._current_joints), np.zeros_like(self._current_joints)

    def get_joint_positions(self, timeout: float = 5.0) -> np.ndarray:
        """Get current joint positions."""
        with self._state_lock:
            return self._current_joints

    def get_base_pose(self, timeout: float = 5.0) -> np.ndarray:
        """Get base pose as [x, y, theta]."""
        with self._state_lock:
            if self._current_pose is None:
                return None
            x, y = self._current_pose['position'][:2]
            theta = self._current_pose['yaw']
            return np.array([x, y, theta])

    # --- Pose Getters ---
    def get_ee_pose(self, matrix=False, link_name=None, q=None):
        """Get end-effector pose."""
        # Simplified version - assumes fixed transform in simulation
        pos = np.array([0.5, 0.0, 1.0])  # Example position
        quat = np.array([0.0, 0.0, 0.0, 1.0])  # Identity quaternion
        if matrix:
            return np.eye(4)  # Return identity matrix for simplicity
        else:
            return pos, quat

    def get_frame_pose(self, q: Union[np.ndarray, dict], node_a: str, node_b: str) -> np.ndarray:
        """Get transform between two frames."""
        # Simplified for simulation - return identity transform
        return np.eye(4)

    def solve_ik(self, pos, quat=None, initial_cfg=None, debug=False, custom_ee_frame=None):
        """Solve inverse kinematics - simplified for simulation."""
        # Return a dummy joint configuration
        return np.array([0.0, 0.5, 0.0, 0.0, 0.0, 0.0])  # Example joint positions

    def _extract_joint_pos(self, q):
        """Extract relevant joint positions from full state."""
        if q is None:
            return None
        # Assuming order: [base_x, lift, arm, yaw, pitch, roll]
        return q[:6] if len(q) >= 6 else q

    # --- Action Commands ---
    def robot_to(self, joint_angles: np.ndarray, blocking: bool = False, timeout: float = 10.0):
        """Send joint position command."""
        msg = Float32MultiArray()
        msg.data = joint_angles.tolist()
        self._joint_cmd_pub.publish(msg)
        if blocking:
            time.sleep(timeout)  # Simplified blocking

    def head_to(self, head_pan, head_tilt, blocking=False, timeout=10.0, reliable=True):
        """Send head position command."""
        # In simulation, we might not have direct head control
        pass  # Could implement using joint commands if needed

    def look_front(self, blocking: bool = True, timeout: float = 10.0):
        """Look forward command."""
        self.head_to(0.0, -0.5, blocking, timeout)  # Example angles

    def look_at_ee(self, blocking: bool = True, timeout: float = 10.0):
        """Look at end-effector command."""
        self.head_to(0.5, -0.3, blocking, timeout)  # Example angles

    def move_base_to(self, xyt, relative=False, blocking=True, timeout=10.0, verbose=False, reliable=True):
        """Move base to specified pose."""
        if relative:
            current_pose = self.get_base_pose()
            xyt = self._relative_to_absolute(xyt, current_pose)
        
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        msg.pose.position.x = xyt[0]
        msg.pose.position.y = xyt[1]
        msg.pose.orientation.z = math.sin(xyt[2] / 2)
        msg.pose.orientation.w = math.cos(xyt[2] / 2)
        
        self._goal_pose_pub.publish(msg)
        if blocking:
            time.sleep(timeout)  # Simplified blocking

    def set_velocity(self, v: float, w: float):
        """Set base velocity."""
        msg = Twist()
        msg.linear.x = v
        msg.angular.z = w
        self._cmd_vel_pub.publish(msg)

    def reset(self):
        """Reset client state."""
        with self._obs_lock:
            self._current_rgb = None
            self._current_depth = None
            self._current_xyz = None
            self._current_camera_info = None
        
        with self._state_lock:
            self._current_pose = None
            self._current_joints = None
            self._at_goal = False

    def move_to_nav_posture(self):
        """Move to navigation posture."""
        # Simplified for simulation - could send specific joint angles
        self.robot_to(np.array([0.0, 0.5, 0.0, 0.0, 0.0, 0.0]))

    # --- State Management ---
    def _update_obs(self, obs):
        """Update observation data."""
        with self._obs_lock:
            # In ROS2, we update through callbacks instead
            pass

    def is_up_to_date(self, no_action=False):
        """Check if state is current."""
        with self._obs_lock:
            return self._current_rgb is not None and self._current_depth is not None

    def _update_pose_graph(self, obs):
        """Update pose graph - not used in basic simulation."""
        pass

    def out_of_date(self):
        """Check if state is outdated."""
        return False  # Simplified for simulation

    def _update_state(self, state: dict) -> None:
        """Update state data."""
        # In ROS2, we update through callbacks instead
        pass

    def at_goal(self) -> bool:
        """Check if at goal."""
        with self._state_lock:
            return self._at_goal

    # --- Map Management ---
    def save_map(self, filename: str):
        """Save map - not implemented in basic simulation."""
        self.get_logger().warn("save_map not implemented in simulation")

    def load_map(self, filename: str):
        """Load map - not implemented in basic simulation."""
        self.get_logger().warn("load_map not implemented in simulation")

    # --- Observation Getters ---
    def get_observation(self, max_iter: int = 5):
        """Get current observation."""
        with self._obs_lock:
            if self._current_rgb is None or self._current_depth is None:
                return None
            
            # Create a simplified observation object
            class SimpleObservation:
                pass
            
            obs = SimpleObservation()
            obs.rgb = self._current_rgb
            obs.depth = self._current_depth
            obs.xyz = self._current_xyz
            obs.camera_K = self._current_camera_info['K'] if self._current_camera_info else None
            return obs

    def get_images(self, compute_xyz=False):
        """Get RGB and depth images."""
        obs = self.get_observation()
        if obs is None:
            return None, None
        if compute_xyz:
            return obs.rgb, obs.depth, obs.xyz
        return obs.rgb, obs.depth

    def get_camera_K(self):
        """Get camera intrinsics."""
        with self._obs_lock:
            return self._current_camera_info['K'] if self._current_camera_info else None

    def get_head_pose(self):
        """Get head pose."""
        # Return identity transform for simulation
        return np.eye(4)

    def set_base_velocity(self, forward: float, rotational: float) -> None:
        """Set base velocity."""
        self.set_velocity(forward, rotational)

    def send_action(self, next_action: Dict[str, Any], timeout: float = 5.0,
                    verbose: bool = False, reliable: bool = True) -> Dict[str, Any]:
        """Send action command."""
        if 'xyt' in next_action:
            self.move_base_to(next_action['xyt'], blocking=next_action.get('nav_blocking', False))
        elif 'v' in next_action and 'w' in next_action:
            self.set_velocity(next_action['v'], next_action['w'])
        elif 'joint' in next_action:
            self.robot_to(np.array(next_action['joint']), blocking=next_action.get('manip_blocking', False))
        return next_action

    # --- Helper Methods ---
    def _relative_to_absolute(self, relative_pose, current_pose):
        """Convert relative pose to absolute."""
        if current_pose is None:
            return relative_pose
        
        x, y, theta = current_pose
        rel_x, rel_y, rel_theta = relative_pose
        
        abs_x = x + rel_x * math.cos(theta) - rel_y * math.sin(theta)
        abs_y = y + rel_x * math.sin(theta) + rel_y * math.cos(theta)
        abs_theta = theta + rel_theta
        
        return np.array([abs_x, abs_y, abs_theta])

    # --- ROS2 Callbacks ---
    def _color_image_callback(self, msg: Image):
        """Callback for color images."""
        try:
            with self._obs_lock:
                self._current_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Error processing color image: {str(e)}")

    def _depth_image_callback(self, msg: Image):
        """Callback for depth images."""
        try:
            with self._obs_lock:
                self._current_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f"Error processing depth image: {str(e)}")

    def _camera_info_callback(self, msg: CameraInfo):
        """Callback for camera info."""
        with self._obs_lock:
            self._current_camera_info = {
                'K': np.array(msg.k).reshape(3, 3),
                'width': msg.width,
                'height': msg.height
            }

    def _point_cloud_callback(self, msg: PointCloud2):
        """Callback for point cloud."""
        try:
            with self._obs_lock:
                gen = point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
                self._current_xyz = np.array(list(gen))
        except Exception as e:
            self.get_logger().error(f"Error processing point cloud: {str(e)}")

    def _odom_callback(self, msg: Odometry):
        """Callback for odometry."""
        with self._state_lock:
            pose = msg.pose.pose
            self._current_pose = {
                'position': [pose.position.x, pose.position.y, pose.position.z],
                'orientation': [pose.orientation.x, pose.orientation.y, 
                               pose.orientation.z, pose.orientation.w],
                'yaw': self._quaternion_to_yaw(pose.orientation)
            }
            
            # Publish TF
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = 'odom'
            t.child_frame_id = 'base_link'
            t.transform.translation.x = pose.position.x
            t.transform.translation.y = pose.position.y
            t.transform.translation.z = pose.position.z
            t.transform.rotation = pose.orientation
            self.tf_broadcaster.sendTransform(t)

    def _joint_states_callback(self, msg: Float32MultiArray):
        """Callback for joint states."""
        with self._state_lock:
            self._current_joints = np.array(msg.data)

    def _quaternion_to_yaw(self, q):
        """Convert quaternion to yaw angle."""
        # yaw (z-axis rotation)
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

def main(args=None):
    rclpy.init(args=args)
    client = DynaMemROS2Client()
    rclpy.spin(client)
    client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()