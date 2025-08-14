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
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

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
# from stretch.visualization.rerun import RerunVisualizer
import timeit
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, QoSHistoryPolicy
from tf2_ros import TransformBroadcaster, Buffer, TransformListener
from rclpy.duration import Duration
from tf2_ros import TransformException
from rclpy.qos_event import SubscriptionEventCallbacks

from rerun_ros2_test import RerunVisualizer


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
        # TF broadcaster already exists; add buffer + listener for lookups
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Optional: gate a one-time warning to avoid log spam if TF is missing
        self._warned_no_cam_tf = False

        
        # Publishers
        self._cmd_vel_pub = self.create_publisher(Twist, '/stretch/cmd_vel', 10)
        self._joint_cmd_pub = self.create_publisher(Float64MultiArray, '/joint_pose_cmd', 10)
        self._goal_pose_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        
        # Common BEST_EFFORT profile for camera topics (matches all publishers)
        camera_qos = QoSProfile(
            depth=1,  # Matches all camera publishers
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST
        )

        # For /odom (matches publisher exactly)
        odom_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST
        )

        # For /joint_states (matches publisher exactly)
        joint_qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST
        )

        self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self._color_image_callback,
            qos_profile=camera_qos
        )
        
        self.create_subscription(
            Image,
            '/camera/depth/image_rect_raw',
            self._depth_image_callback,
            qos_profile=camera_qos
        )
        
        self.create_subscription(
            CameraInfo,
            '/camera/color/camera_info',
            self._camera_info_callback,
            qos_profile=camera_qos
        )
        
        self.create_subscription(
            PointCloud2,
            '/camera/depth/color/points',
            self._point_cloud_callback,
            qos_profile=camera_qos
        )
        
        self.create_subscription(
            Odometry,
            '/odom',
            self._odom_callback,
            qos_profile=odom_qos
        )
        
        self.create_subscription(
            JointState,
            '/stretch/joint_states',
            # '/joint_states',
            self._joint_states_callback,
            qos_profile=joint_qos
        )

        self.get_logger().info("Subscription Summary:")
        topics = [
            '/camera/color/image_raw',
            '/camera/depth/image_rect_raw',
            '/camera/color/camera_info',
            '/camera/depth/color/points',
            '/odom',
            '/joint_states'
        ]
        for topic in topics:
            self.get_logger().info(f"{topic}: {self.count_subscribers(topic)} subscribers")
        # breakpoint()
        # self._odom_subs = self._subscribe_state_topic_dual_qos(
        #     Odometry, '/odom', self._odom_callback
        # )
        # self._joints_subs = self._subscribe_state_topic_dual_qos(
        #     JointState, '/joint_states', self._joint_states_callback
        # )
        self._got_first_odom = False
        self._got_first_joints = False


        self._robot_model = self
        self.dof = 3 + 2 + 4 + 2
        self.xyt = np.zeros(3)


        self._started = False
        enable_rerun_server = True
        output_path = None
        # Optional live visualization
        if enable_rerun_server:
            self._rerun = RerunVisualizer(output_path=output_path)
        else:
            self._rerun = None
            self._rerun_thread = None
        
        self._is_homed = True         # In sim, assume homed unless told otherwise
        self._is_runstopped = False   # In sim, assume not runstopped

        self.get_logger().info("DynaMem ROS2 client initialized")


    def _subscribe_state_topic_dual_qos(self, msg_type, topic_name, callback):
        """
        Create two subscriptions on the same topic with different QoS profiles:
        one RELIABLE and one BEST_EFFORT. Whichever matches the publisher will win.
        """
        # Debug callbacks for QoS issues
        def _incompatible_qos_cb(event):
            self.get_logger().warn(
                f"[QoS] Incompatible QoS on {topic_name}: "
                f"total_count={event.total_count} change={event.total_count_change}"
            )

        event_cbs = SubscriptionEventCallbacks(incompatible_qos=_incompatible_qos_cb)

        # RELIABLE profile
        qos_rel = QoSProfile(depth=10,
                            reliability=QoSReliabilityPolicy.RELIABLE,
                            durability=QoSDurabilityPolicy.VOLATILE)

        # BEST_EFFORT profile
        qos_be = QoSProfile(depth=10,
                            reliability=QoSReliabilityPolicy.BEST_EFFORT,
                            durability=QoSDurabilityPolicy.VOLATILE)

        sub_rel = self.create_subscription(
            msg_type, topic_name, callback, qos_profile=qos_rel, event_callbacks=event_cbs
        )
        sub_be = self.create_subscription(
            msg_type, topic_name, callback, qos_profile=qos_be, event_callbacks=event_cbs
        )

        return sub_rel, sub_be


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
    
    # ---- lifecycle ----
    def is_homed(self) -> bool:
        """Check if robot is homed."""
        with self._state_lock:
            return self._is_homed

    def is_runstopped(self) -> bool:
        """Check if robot is in runstop state."""
        with self._state_lock:
            return self._is_runstopped

    def start(self):
        """Start one executor thread + light worker threads; wait for initial data."""
        if self._started:
            return True

        self.get_logger().info("Starting DynaMem ROS2 client...")
        self._finish = False

        # SINGLE executor, single place that spins
        self._executor = rclpy.executors.MultiThreadedExecutor()
        self._executor.add_node(self)

        def _spin_forever():
            while rclpy.ok() and not self._finish:
                try:
                    self._executor.spin_once(timeout_sec=0.1)
                except Exception as e:
                    self.get_logger().error(f"Executor spin error: {e}")
                    break

        self._executor_thread = threading.Thread(target=_spin_forever, daemon=True)
        self._executor_thread.start()

        # Workers: no spinning—just periodic processing
        self._obs_thread = threading.Thread(target=self.blocking_spin, kwargs={"verbose": False, "visualize": False}, daemon=True)
        self._state_thread = threading.Thread(target=self.blocking_spin_state, kwargs={"verbose": False}, daemon=True)
        self._servo_thread = threading.Thread(target=self.blocking_spin_servo, kwargs={"verbose": False}, daemon=True)
        self._obs_thread.start()
        self._state_thread.start()
        self._servo_thread.start()

        if self._rerun:
            self._rerun_thread = threading.Thread(target=self.blocking_spin_rerun, daemon=True)

            self._rerun_thread.start()

        # Wait up to 10s for first obs+state
        t0 = timeit.default_timer()
        last_print = 0.0

        # Track first-seen times for each piece of data
        first_seen = {"rgb": None, "depth": None, "joints": None, "pose": None}

        while rclpy.ok() and not self._finish:
            # Snapshot under locks
            with self._obs_lock:
                rgb_ready   = self._current_rgb is not None
                depth_ready = self._current_depth is not None
            with self._state_lock:
                joints_ready = self._current_joints is not None
                pose_ready   = self._current_pose is not None

            # Record first-seen times
            now = timeit.default_timer()
            if rgb_ready   and first_seen["rgb"]   is None: first_seen["rgb"]   = now - t0
            if depth_ready and first_seen["depth"] is None: first_seen["depth"] = now - t0
            if joints_ready and first_seen["joints"] is None: first_seen["joints"] = now - t0
            if pose_ready   and first_seen["pose"]   is None: first_seen["pose"]   = now - t0

            obs_ready = rgb_ready and depth_ready
            state_ready = joints_ready and pose_ready

            if obs_ready and state_ready:
                elapsed = now - t0
                self.get_logger().info(
                    f"Initial data ready in {elapsed:.2f}s "
                    f"(rgb@{first_seen['rgb']:.2f}s, depth@{first_seen['depth']:.2f}s, "
                    f"joints@{first_seen['joints']:.2f}s, pose@{first_seen['pose']:.2f}s)"
                )
                break

            # Heartbeat debug every ~0.5s
            if (now - last_print) > 0.5:
                missing = []
                if not rgb_ready:    missing.append("rgb")
                if not depth_ready:  missing.append("depth")
                if not joints_ready: missing.append("joints")
                if not pose_ready:   missing.append("pose")

                elapsed = now - t0
                remaining = max(0.0, 10.0 - elapsed)
                seen_str = (
                    f"seen: rgb={rgb_ready}, depth={depth_ready}, "
                    f"joints={joints_ready}, pose={pose_ready}"
                )
                firsts = ", ".join(
                    f"{k}={('%.2fs' % v) if v is not None else '-'}" for k, v in first_seen.items()
                )

                self.get_logger().info(
                    f"[startup] waiting for initial observations/state | "
                    f"missing={missing} | {seen_str} | first_seen: {firsts} | "
                    f"elapsed={elapsed:.2f}s remaining={remaining:.2f}s"
                )
                last_print = now

            if now - t0 > 10.0:
                self.get_logger().error("Timeout waiting for initial observations/state")
                self.get_logger().info(
                    "Check publishers for: "
                    f"{'rgb ' if not rgb_ready else ''}"
                    f"{'depth ' if not depth_ready else ''}"
                    f"{'joints ' if not joints_ready else ''}"
                    f"{'pose ' if not pose_ready else ''}".strip()
                )
                return False

            time.sleep(0.1)


        if not self.is_homed():
            self.stop()
            raise RuntimeError("Robot not homed; please home before running.")
        if self.is_runstopped():
            self.stop()
            raise RuntimeError("Robot runstopped; release the runstop button.")

        self._started = True
        return True

    # ---- worker loops (no spinning here) ----
    def blocking_spin_rerun(self) -> None:
        self.get_logger().info("blocking_spin_rerun")
        while rclpy.ok() and not self._finish:
            with self._obs_lock:
                rgb = self._current_rgb
                depth = self._current_depth
                xyz = self._current_xyz
                K = self._current_camera_info['K'] if self._current_camera_info else None

            if rgb is not None and depth is not None and xyz is not None:
                obs = {'rgb': rgb, 'depth': depth, 'xyz': xyz, 'camera_K': K}
                try:
                    print("rerun step")
                    self._rerun.step(obs, None)
                except Exception as e:
                    self.get_logger().error(f"Error in rerun visualization: {str(e)}")
            time.sleep(0.1)

    def blocking_spin_state(self, verbose: bool = False):
        self.get_logger().info("blocking_spin_state")
        while rclpy.ok() and not self._finish:
            try:
                if verbose:
                    with self._state_lock:
                        if self._current_joints is not None:
                            self.get_logger().info(f"Current joints: {self._current_joints}")
                time.sleep(0.1)
            except Exception as e:
                self.get_logger().error(f"Error in state processing: {str(e)}")
                break

    def blocking_spin_servo(self, verbose: bool = False):
        self.get_logger().info("blocking_spin_servo")
        while rclpy.ok() and not self._finish:
            try:
                time.sleep(0.1)
            except Exception as e:
                self.get_logger().error(f"Error in servo processing: {str(e)}")
                break

    def blocking_spin(self, verbose: bool = False, visualize: bool = False):
        """Main observation loop—reads the latest data, does light processing."""
        self.get_logger().info("blocking_spin")
        sum_time = 0.0
        steps = 0
        t0 = timeit.default_timer()
        shown_point_cloud = visualize

        while rclpy.ok() and not self._finish:
            try:
                # Snapshot under lock
                with self._obs_lock:
                    rgb = self._current_rgb
                    depth = self._current_depth
                    xyz = self._current_xyz
                    cam_info = self._current_camera_info

                if rgb is None or depth is None or cam_info is None:
                    time.sleep(0.1)
                    continue

                output = {
                    'rgb': rgb,
                    'depth': depth,
                    'xyz': xyz,
                    'camera_K': cam_info['K'],
                    'rgb_height': cam_info['height'],
                    'rgb_width': cam_info['width'],
                    'step': steps
                }

                # optional one-time point cloud visualization could go here (outside the lock)

                self._update_obs(output)
                self._update_pose_graph(output)

                # timing
                t1 = timeit.default_timer()
                dt = t1 - t0
                sum_time += dt
                steps += 1
                if verbose:
                    self.get_logger().info(f"Control mode: {self._control_mode}")
                    self.get_logger().info(f"time taken = {dt:.4f}s avg = {sum_time/steps:.4f}s")
                t0 = timeit.default_timer()

                time.sleep(0.1)

            except Exception as e:
                self.get_logger().error(f"Error in blocking_spin: {str(e)}")
                break

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
        msg = Float64MultiArray()
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
        
            # Do TF lookup OUTSIDE the lock
            cam_T = self._lookup_tf_matrix("odom", "camera_color_frame", timeout_sec=0.5)
            obs.camera_pose = cam_T  # 4x4 np.ndarray or None if not available

            return obs
    
    def _transform_to_matrix(self, t: TransformStamped) -> np.ndarray:
        """Convert a geometry_msgs/TransformStamped into a 4x4 homogeneous matrix."""
        tx = t.transform.translation.x
        ty = t.transform.translation.y
        tz = t.transform.translation.z
        qx = t.transform.rotation.x
        qy = t.transform.rotation.y
        qz = t.transform.rotation.z
        qw = t.transform.rotation.w

        # rotation matrix from quaternion
        x2, y2, z2 = qx*qx, qy*qy, qz*qz
        xy, xz, yz = qx*qy, qx*qz, qy*qz
        wx, wy, wz = qw*qx, qw*qy, qw*qz

        R = np.array([
            [1 - 2*(y2 + z2),     2*(xy - wz),         2*(xz + wy)],
            [2*(xy + wz),         1 - 2*(x2 + z2),     2*(yz - wx)],
            [2*(xz - wy),         2*(yz + wx),         1 - 2*(x2 + y2)]
        ], dtype=np.float64)

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = [tx, ty, tz]
        return T

    def _lookup_tf_matrix(self, target_frame: str, source_frame: str, timeout_sec: float = 0.2) -> Optional[np.ndarray]:
        """
        Return 4x4 pose of `source_frame` in `target_frame` coordinates, or None if not available.
        Equivalent to: target_T_source.
        """
        try:
            t = self.tf_buffer.lookup_transform(
                target_frame,          # to frame
                source_frame,          # from frame
                rclpy.time.Time(),     # latest available
                timeout=Duration(seconds=timeout_sec)
            )
            print("t: ", t)
            breakpoint()
            return self._transform_to_matrix(t)
        except TransformException as e:
            if not self._warned_no_cam_tf:
                self.get_logger().warn(f"TF lookup failed ({target_frame} <- {source_frame}): {e}")
                self._warned_no_cam_tf = True
            return None


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
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            with self._obs_lock:
                self._current_rgb = img
        except Exception as e:
            self.get_logger().error(f"Error processing color image: {str(e)}")

    def _depth_image_callback(self, msg: Image):
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            with self._obs_lock:
                self._current_depth = depth
        except Exception as e:
            self.get_logger().error(f"Error processing depth image: {str(e)}")

    def _camera_info_callback(self, msg: CameraInfo):
        info = {
            'K': np.array(msg.k).reshape(3, 3),
            'width': msg.width,
            'height': msg.height
        }
        with self._obs_lock:
            self._current_camera_info = info

    def _point_cloud_callback(self, msg: PointCloud2):
        try:
            gen = point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
            xyz = np.array(list(gen))
            with self._obs_lock:
                self._current_xyz = xyz
        except Exception as e:
            self.get_logger().error(f"Error processing point cloud: {str(e)}")

    def _odom_callback(self, msg: Odometry):
        try:
            print(f"Raw Odom Received! Position: {msg.pose.pose.position}")  # Immediate verification
            pose = msg.pose.pose
            yaw = self._quaternion_to_yaw(pose.orientation)
            # with self._state_lock:  # Temporarily remove this for testing
            self._current_pose = {
                'position': [pose.position.x, pose.position.y, pose.position.z],
                'orientation': [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w],
                'yaw': yaw
            }
            print(f"Processed Odom: {self._current_pose}")  # Verify processing
        except Exception as e:
            print(f"Odom callback error: {e}")

    def _joint_states_callback(self, msg: JointState):
        try:
            print(f"Raw Joint States Received! Names: {msg.name} Positions: {msg.position}")
            # with self._state_lock:  # Temporarily remove this for testing
            self._current_joints = np.array(msg.position)
            print(f"Processed Joints: {self._current_joints}")
        except Exception as e:
            print(f"Joint states callback error: {e}")

    # def _odom_callback(self, msg: Odometry):
    #     pose = msg.pose.pose
    #     print("odom callback: ", pose)
    #     yaw = self._quaternion_to_yaw(pose.orientation)
    #     with self._state_lock:
    #         self._current_pose = {
    #             'position': [pose.position.x, pose.position.y, pose.position.z],
    #             'orientation': [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w],
    #             'yaw': yaw
    #         }

    #     # Publish TF outside the lock
    #     t = TransformStamped()
    #     t.header.stamp = self.get_clock().now().to_msg()
    #     t.header.frame_id = 'odom'
    #     t.child_frame_id = 'base_link'
    #     t.transform.translation.x = pose.position.x
    #     t.transform.translation.y = pose.position.y
    #     t.transform.translation.z = pose.position.z
    #     t.transform.rotation = pose.orientation
    #     self.tf_broadcaster.sendTransform(t)

    # def _joint_states_callback(self, msg: JointState):
    #     position = msg.position
    #     print("joint states callback: ", position)
    #     with self._state_lock:
    #         self._current_joints = np.array(position)

    # def _odom_callback(self, msg: Odometry):
    #     if not self._got_first_odom:
    #         self.get_logger().info("[startup] got first /odom")
    #         self._got_first_odom = True
    #     pose = msg.pose.pose
    #     yaw = self._quaternion_to_yaw(pose.orientation)
    #     with self._state_lock:
    #         self._current_pose = {
    #             'position': [pose.position.x, pose.position.y, pose.position.z],
    #             'orientation': [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w],
    #             'yaw': yaw
    #         }
    #     # Publish TF (as you had)
    #     t = TransformStamped()
    #     t.header.stamp = self.get_clock().now().to_msg()
    #     t.header.frame_id = 'odom'
    #     t.child_frame_id = 'base_link'
    #     t.transform.translation.x = pose.position.x
    #     t.transform.translation.y = pose.position.y
    #     t.transform.translation.z = pose.position.z
    #     t.transform.rotation = pose.orientation
    #     self.tf_broadcaster.sendTransform(t)

    # def _joint_states_callback(self, msg: JointState):
    #     if not self._got_first_joints:
    #         self.get_logger().info("[startup] got first /joint_states")
    #         self._got_first_joints = True
    #     with self._state_lock:
    #         self._current_joints = np.array(msg.position)



    def _quaternion_to_yaw(self, q):
        """Convert quaternion to yaw angle."""
        # yaw (z-axis rotation)
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

# def main(args=None):
#     rclpy.init(args=args)
#     client = DynaMemROS2Client()
#     rclpy.spin(client)
#     client.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()