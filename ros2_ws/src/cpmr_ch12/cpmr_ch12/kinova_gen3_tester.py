from tokenize import String
from kinova_gen3_interfaces.srv import Status, SetGripper, GetGripper, SetJoints, GetJoints, GetTool, SetTool
import rclpy
from rclpy.node import Node
import time
from std_msgs.msg import Bool
import json


def do_home(node, home):
    z = Status.Request()
    future = home.call_async(z)
    rclpy.spin_until_future_complete(node, future)
    print(f"Home returns {future.result()}")
    return future.result().status

def do_set_gripper(node, set_gripper, v):
    """Set the gripper"""
    z = SetGripper.Request()
    z.value = v
    future = set_gripper.call_async(z)
    rclpy.spin_until_future_complete(node, future)
    print(f"SetGripper returns {future.result()}")
    return future.result().status

def do_get_gripper(node, get_gripper):
    """Get the current gripper setting"""
    z = GetGripper.Request()
    future = get_gripper.call_async(z)
    rclpy.spin_until_future_complete(node, future)
    print(f"GetGripper returns {future.result()}")
    return future.result().value

def do_get_tool(node, get_tool):
    z = GetTool.Request()
    future = get_tool.call_async(z)
    rclpy.spin_until_future_complete(node, future)
    print(f"GetTool returns {future.result()}")
    q = future.result()
    return q.x, q.y, q.z, q.theta_x, q.theta_y, q.theta_z

def do_set_tool(node, set_tool, x, y, z, theta_x, theta_y, theta_z):
    t = SetTool.Request()
    t.x = float(x)
    t.y = float(y)
    t.z = float(z)
    t.theta_x = float(theta_x)
    t.theta_y = float(theta_y)
    t.theta_z = float(theta_z)
    print(f"Request built {t}")
    future = set_tool.call_async(t)
    rclpy.spin_until_future_complete(node, future)
    print(f"SetTool returns {future.result()}")
    return future.result().status

def do_get_joints(node, get_joints):
    z = GetJoints.Request()
    future = get_joints.call_async(z)
    rclpy.spin_until_future_complete(node, future)
    print(f"GetJoints returns {future.result()}")
    return future.result().joints

def do_set_joints(node, set_joints, v):
    z = SetJoints.Request(joints=v)
    print(f"Request built {z}")
    future = set_joints.call_async(z)
    rclpy.spin_until_future_complete(node, future)
    print(f"SetJoints returns {future.result()}")
    return future.result().status

def movement(node, set_tool, left_above, right_above, right_below, left_below):

    if left_above:
        x = 0.3
        y = 0.2
        z = 0.2
    elif right_above:
        x = 0.3
        y = -0.2
        z = 0.2
    elif right_below:
        x = 0.5
        y = -0.2
        z = 0.0
    elif left_below:
        x = 0.5
        y = 0.2
        z = 0.0
    else:
        return False
    

    # Move arm
    do_set_tool(node, set_tool, x, y, z, 180.0, 0.0, 180.0)  # Move above block
    time.sleep(1.5)

    return True

class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')
        
        # Create service clients
        self.get_tool = self.create_client(GetTool, "/get_tool")
        while not self.get_tool.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for get_tool')

        self.set_tool = self.create_client(SetTool, "/set_tool")
        while not self.set_tool.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for set_tool')

        self.get_joints = self.create_client(GetJoints, "/get_joints")
        while not self.get_joints.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for get_joints')

        self.set_joints = self.create_client(SetJoints, "/set_joints")
        while not self.set_joints.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for set_joints')

        self.set_gripper = self.create_client(SetGripper, "/set_gripper")
        while not self.set_gripper.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for set_gripper')

        self.get_gripper = self.create_client(GetGripper, "/get_gripper")
        while not self.get_gripper.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for get_gripper')

        self.home = self.create_client(Status, "/home")
        while not self.home.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for home')

        # Create subscription
        # self.subscription = self.create_subscription(
        #     Bool,  # Adjust message type based on your actual topic
        #     '/pose_keypoints',
        #     self.pose_callback,
        #     10)
        
        self.subscription = self.create_subscription(
            String,               # Message type from yolo_pose
            '/pose_keypoints',    # Topic name
            self.pose_callback,   # Callback function
            10
        )

        def pose_callback(self, msg):
            # Parse JSON string from yolo_pose
            try:
                pose_flags = json.loads(msg.data)
                left_above = pose_flags.get('left_above', False)
                left_below = pose_flags.get('left_below', False)
                right_above = pose_flags.get('right_above', False)
                right_below = pose_flags.get('right_below', False)

                self.get_logger().info(
                    f"Received flags - Left Above: {left_above}, Left Below: {left_below}, "
                    f"Right Above: {right_above}, Right Below: {right_below}"
                )

            except Exception as e:
                self.get_logger().error(f"Failed to parse pose flags: {e}")

# def pose_callback(self, msg):
#         left_above = msg.data[0] if hasattr(msg, 'data') and len(msg.data) > 0 else False
#         left_below = msg.data[1] if hasattr(msg, 'data') and len(msg.data) > 1 else False
#         right_above = msg.data[2] if hasattr(msg, 'data') and len(msg.data) > 2 else False
#         right_below = msg.data[3] if hasattr(msg, 'data') and len(msg.data) > 3 else False
        
#         movement(self, self.set_tool, left_above, right_above, right_below, left_below)


def main():
    rclpy.init(args=None)
    node = RobotController()
    
    # Test movement
    # movement(node, node.set_tool, True, False, False, False)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

    # # Spin to keep the node alive and process callbacks
    # rclpy.spin(node)
    
    # node.destroy_node()
    # rclpy.shutdown()

if __name__ == '__main__':
    main()
