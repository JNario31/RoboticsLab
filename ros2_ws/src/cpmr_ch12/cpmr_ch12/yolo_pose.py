import os
import sys
import rclpy
import cv2
import datetime
import numpy as np
import pandas as pd
import math
import threading

from rclpy.node import Node
from ultralytics import YOLO
from cv_bridge import CvBridge
from std_srvs.srv import SetBool
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from rclpy.qos import qos_profile_sensor_data
from ultralytics.engine.results import Results, Keypoints
from ament_index_python.packages import get_package_share_directory

from kinova_gen3_interfaces.srv import Status, SetGripper, GetGripper, SetJoints, GetJoints, GetTool, SetTool

class YOLO_Pose(Node):
    _BODY_PARTS = ["NOSE", "LEFT_EYE", "RIGHT_EYE", "LEFT_EAR", "RIGHT_EAR", "LEFT_SHOULDER", "RIGHT_SHOULDER",
                   "LEFT_ELBOW", "RIGHT_ELBOW", "LEFT_WRIST", "RIGHT_WRIST", "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE",
                   "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE"]
    
    def __init__(self):
        super().__init__('pose_node')
        self._i = 1
        self._moving = False  # Flag to prevent multiple concurrent movements
        
        # params
        self._model_file = os.path.join(get_package_share_directory('cpmr_ch12'), 'yolov8n-pose.pt') 
        self.declare_parameter("model", self._model_file) 
        model = self.get_parameter("model").get_parameter_value().string_value

        self.declare_parameter("device", "cpu")
        self._device = self.get_parameter("device").get_parameter_value().string_value

        self.declare_parameter("threshold", 0.5)
        self._threshold = self.get_parameter("threshold").get_parameter_value().double_value

        self.declare_parameter("camera_topic", "/mycamera/image_raw")
        self._camera_topic = self.get_parameter("camera_topic").get_parameter_value().string_value

        self._move_flag = False
        self._bridge = CvBridge()
        self._model = YOLO(model)
        self._model.fuse()

        # Create service clients (not services!)
        self._home_client = self.create_client(Status, "/home")
        self._set_tool_client = self.create_client(SetTool, "/set_tool")
        self._get_tool_client = self.create_client(GetTool, "/get_tool")
        self._set_joints_client = self.create_client(SetJoints, "/set_joints")
        self._get_joints_client = self.create_client(GetJoints, "/get_joints")
        self._set_gripper_client = self.create_client(SetGripper, "/set_gripper")
        self._get_gripper_client = self.create_client(GetGripper, "/get_gripper")

        # Wait for services
        self.get_logger().info('Waiting for Kinova services...')
        self._home_client.wait_for_service()
        self._set_tool_client.wait_for_service()
        self.get_logger().info('Kinova services available')

        # subs
        self._sub = self.create_subscription(Image, self._camera_topic, self._camera_callback, 1) 

    def parse_keypoints(self, results: Results):
        keypoints_list = []

        for points in results.keypoints:        
            if points.conf is None:
                continue

            for kp_id, (p, conf) in enumerate(zip(points.xy[0], points.conf[0])):
                if conf >= self._threshold:
                    keypoints_list.append([kp_id, p[0], p[1], conf])

        return keypoints_list
    
    def _camera_callback(self, data):
        self.get_logger().info(f'{self.get_name()} camera callback')
        img = self._bridge.imgmsg_to_cv2(data)
        results = self._model.predict(
                source = img,
                verbose = False,
                stream = False,
                conf = self._threshold,
                device = self._device
        )

        if len(results) != 1:
            self.get_logger().info(f'{self.get_name()}  Nothing to see here or too much {len(results)}')
            return
            
        results = results[0].cpu()
        if len(results.boxes.data) == 0:
            self.get_logger().info(f'{self.get_name()}  boxes are too small')
            return

        if results.keypoints:
            keypoints = self.parse_keypoints(results)
            left_shoulder = None
            right_shoulder = None
            if len(keypoints) > 0:
                # Visualize results on frame        
                annotated_frame = results[0].plot()
                cv2.imshow('Results', annotated_frame)
                cv2.waitKey(1)
        
            key_dict = {YOLO_Pose._BODY_PARTS[kp[0]]: kp for kp in keypoints}

            required = ["LEFT_EYE", "RIGHT_EYE", "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_WRIST", "RIGHT_WRIST"]
            if not all(k in key_dict for k in required):
                return

            left_eye_y = key_dict["LEFT_EYE"][2]
            right_eye_y = key_dict["RIGHT_EYE"][2]
            left_shoulder_y = key_dict["LEFT_SHOULDER"][2]
            right_shoulder_y = key_dict["RIGHT_SHOULDER"][2]
            left_wrist_y = key_dict["LEFT_WRIST"][2]
            right_wrist_y = key_dict["RIGHT_WRIST"][2]

            dy_ref = abs((left_eye_y + right_eye_y)/2 - (left_shoulder_y + right_shoulder_y)/2)

            left_above = left_wrist_y < (left_shoulder_y - dy_ref)
            left_below = left_wrist_y > (left_shoulder_y + dy_ref)
            right_above = right_wrist_y < (right_shoulder_y - dy_ref)
            right_below = right_wrist_y > (right_shoulder_y + dy_ref)

            self.get_logger().info(f'{self.get_name()}  Left Above: {left_above}, Left Below: {left_below}, Right Above: {right_above}, Right Below: {right_below}')
           
            if not self._moving:
                if left_above:
                    self.get_logger().info('LEFT HAND ABOVE - Moving to position 1')
                    self._moving = True
                    self.call_set_tool_async(0.1, 0.2, 0.1, 180.0, 0.0, 180.0)
                    
                elif left_below:
                    self.get_logger().info('LEFT HAND BELOW - Moving to position 2')
                    self._moving = True
                    self.call_set_tool_async(-0.1, 0.2, 0.1, 180.0, 0.0, 180.0)
                    
                elif right_above:
                    self.get_logger().info('RIGHT HAND ABOVE - Moving to position 3')
                    self._moving = True
                    self.call_set_tool_async(0.0, 0.1, 0.1, 180.0, 0.0, 180.0)
                    
                elif right_below:
                    self.get_logger().info('RIGHT HAND BELOW - Moving to position 4')
                    self._moving = True
                    self.call_set_tool_async(0.0, 0.3, 0.1, 180.0, 0.0, 180.0)

    def call_home(self):
        """Call home by moving to a specific Cartesian position"""
        self.get_logger().info('Moving to home position...')
        
        # Define your custom home position in Cartesian space
        # x, y, z in meters, angles in degrees
        home_x = 0.4
        home_y = 0.0
        home_z = 0.4
        home_theta_x = 180.0
        home_theta_y = 0.0
        home_theta_z = 90.0
        
        return self.call_set_tool(home_x, home_y, home_z, 
                                home_theta_x, home_theta_y, home_theta_z)

    def call_set_tool(self, x, y, z, theta_x, theta_y, theta_z):
        """Call the set_tool service (blocking - use in main, not in callbacks)"""
        request = SetTool.Request()
        request.x = float(x)
        request.y = float(y)
        request.z = float(z)
        request.theta_x = float(theta_x)
        request.theta_y = float(theta_y)
        request.theta_z = float(theta_z)
        
        future = self._set_tool_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        
        if future.result() is not None:
            self.get_logger().info(f'SetTool returned: {future.result().status}')
            return future.result().status
        else:
            self.get_logger().error('SetTool service call failed')
            return False

    def call_set_tool_async(self, x, y, z, theta_x, theta_y, theta_z):
        """Call the set_tool service asynchronously (safe for callbacks)"""
        request = SetTool.Request()
        request.x = float(x)
        request.y = float(y)
        request.z = float(z)
        request.theta_x = float(theta_x)
        request.theta_y = float(theta_y)
        request.theta_z = float(theta_z)
        
        future = self._set_tool_client.call_async(request)
        future.add_done_callback(self._set_tool_callback)

    def _set_tool_callback(self, future):
        """Callback when set_tool service completes"""
        try:
            result = future.result()
            self.get_logger().info(f'SetTool completed: {result.status}')
            self._moving = False
        except Exception as e:
            self.get_logger().error(f'SetTool failed: {str(e)}')
            self._moving = False


def main(args=None):
    rclpy.init(args=args)
    node = YOLO_Pose()

    # Call home at startup (blocking call is OK here in main)
    # success = node.call_home()
    # if success:
    #     node.get_logger().info('Robot homed successfully')
    # else:
    #     node.get_logger().error('Failed to home robot')

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()