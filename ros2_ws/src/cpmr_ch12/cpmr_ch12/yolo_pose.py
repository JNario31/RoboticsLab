import os
import sys
from tokenize import String
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

class YOLO_Pose(Node):
    _BODY_PARTS = ["NOSE", "LEFT_EYE", "RIGHT_EYE", "LEFT_EAR", "RIGHT_EAR", "LEFT_SHOULDER", "RIGHT_SHOULDER",
                   "LEFT_ELBOW", "RIGHT_ELBOW", "LEFT_WRIST", "RIGHT_WRIST", "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE",
                   "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE"]
    def __init__(self):
        super().__init__('pose_node')

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

        # publisher
        self._keypoint_pub = self.create_publisher(bool, '/pose_keypoints', 10)

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
        #self.get_logger().info(f'{self.get_name()} camera callback')
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

        key_dict = {YOLO_Pose._BODY_PARTS[k[0]]: k for k in keypoints}
         # === NEW CODE: Check if needed keypoints exist ===
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

        self.get_logger().info(f'{left_above} {left_below} {right_above} {right_below}')
        keypointArray = [left_above, left_below, right_above, right_below]
        self._keypoint_pub.publish(keypointArray)

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