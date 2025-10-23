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

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2, Common_pb2

from cpmr_ch12.utilities import parseConnectionArguments, DeviceConnection

from kinova_gen3_node import Kinova_Gen3_Interface

class YOLO_Pose(Node):
    _BODY_PARTS = ["NOSE", "LEFT_EYE", "RIGHT_EYE", "LEFT_EAR", "RIGHT_EAR", "LEFT_SHOULDER", "RIGHT_SHOULDER",
                   "LEFT_ELBOW", "RIGHT_ELBOW", "LEFT_WRIST", "RIGHT_WRIST", "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE",
                   "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE"]
    def __init__(self):
        super().__init__('pose_node')

        self._kinova_interface = Kinova_Gen3_Interface()
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

        # subs
        self._sub = self.create_subscription(Image, self._camera_topic, self._camera_callback, 1) 

        # Create the Kinova Gen3 interface object
        self.create_service(Status, "home", self._handle_home)
        self.create_service(SetJoints, "set_joints", self._handle_set_joints)
        self.create_service(GetJoints, "get_joints", self._handle_get_joints)
        self.create_service(SetTool, "set_tool", self._handle_set_tool)
        self.create_service(GetTool, "get_tool", self._handle_get_tool)

        args = parseConnectionArguments()
        with DeviceConnection.createTcpConnection(args) as router:
            self._router = router
            self._base = BaseClient(self._router)
            self._base_cyclic = BaseCyclicClient(self._router)

        if example_move_to_home_position(self._base):
           self.get_logger().info('Robot initialized successfully')
        else:
           self.get_logger().error('Failed to initialize robot position')
        
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
                for i in range(len(keypoints)):
                    self.get_logger().info(f'{self.get_name()}  {YOLO_Pose._BODY_PARTS[keypoints[i][0]]} {keypoints[i]}')

                # Visualize results on frame        
                annotated_frame = results[0].plot()
                cv2.imshow('Results', annotated_frame)
                cv2.waitKey(1)

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
            """Call the set_tool service"""
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

def main(args=None):
    rclpy.init(args=args)
    node = YOLO_Pose()

    try:
        rclpy.spin(node)
        rclpy.shutdown()
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()