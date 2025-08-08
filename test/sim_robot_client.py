import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import numpy as np
import tf_transformations
import sys
# sys.path.append("/home/koyo/stretch_ai/src")
# from stretch.core.interfaces import Observations
# from stretch.core.robot import AbstractRobotClient


class SimRobotClient():
    def __init__(self):
        self.rgb = None
        self.depth = None
        self.K = None
        self.camera_pose = np.eye(4)
        self.base_pose = np.zeros(3)

        self.bridge = CvBridge()

        rclpy.init()
        self.node = rclpy.create_node("sim_robot_client")

        self.node.create_subscription(Image, "/camera/color/image_raw", self._rgb_cb, 10)
        self.node.create_subscription(Image, "/camera/depth/image_rect_raw", self._depth_cb, 10)
        self.node.create_subscription(CameraInfo, "/camera/color/camera_info", self._info_cb, 10)
        self.node.create_subscription(Odometry, "/odom", self._odom_cb, 10)

        self.goal_pub = self.node.create_publisher(PoseStamped, "/goal_pose", 10)

        self._control_mode = "navigation"

    def _rgb_cb(self, msg):
        self.rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")

    def _depth_cb(self, msg):
        self.depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough") / 1000.0

    def _info_cb(self, msg):
        self.K = np.array(msg.k).reshape(3, 3)

    def _odom_cb(self, msg):
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        yaw = tf_transformations.euler_from_quaternion([ori.x, ori.y, ori.z, ori.w])[2]
        self.base_pose = np.array([pos.x, pos.y, yaw])

    def get_observation(self):
        if self.rgb is None or self.depth is None or self.K is None:
            return None

        return Observations(
            rgb=self.rgb,
            depth=self.depth,
            camera_K=self.K,
            camera_pose=self.camera_pose,
            gps=self.base_pose[:2],
            compass=self.base_pose[2:],
        )

    def get_base_pose(self):
        return self.base_pose

    def move_base_to(self, xyt, blocking=True, **kwargs):
        msg = PoseStamped()
        msg.header.frame_id = "map"
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.pose.position.x = float(xyt[0])
        msg.pose.position.y = float(xyt[1])
        quat = tf_transformations.quaternion_from_euler(0, 0, float(xyt[2]))
        msg.pose.orientation.x = quat[0]
        msg.pose.orientation.y = quat[1]
        msg.pose.orientation.z = quat[2]
        msg.pose.orientation.w = quat[3]
        self.goal_pub.publish(msg)
        print(f"[SimRobotClient] Published goal: {xyt}")

    def look_front(self, **kwargs):
        print("[SimRobotClient] Looking forward — simulated.")

    def switch_to_navigation_mode(self, **kwargs):
        print("[SimRobotClient] Switched to navigation mode.")
        self._control_mode = "navigation"

    def move_to_nav_posture(self):
        print("[SimRobotClient] Simulated move to nav posture.")

    def say(self, text: str):
        print(f"[SimRobotClient] (say): {text}")

    def stop(self):
        print("[SimRobotClient] Shutting down.")
        self.node.destroy_node()
        rclpy.shutdown()
