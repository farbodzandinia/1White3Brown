#!/usr/bin/env python3

import rospy
import numpy as np
import cv2 as cv
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
import tensorflow as tf
from gazebo_msgs.srv import SetModelState, SetModelStateRequest
from std_msgs.msg import String

class RobotDriver:
    def __init__(self, model_paths):
        """
        Initialize the robot driver with TensorFlow Lite models for asphalt and desert environments,
        set up ROS subscribers and publishers, and define thresholds for color detection.
        """
        # Load TensorFlow Lite models and allocate tensors
        self.interpreter_asphalt = tf.lite.Interpreter(model_paths['asphalt'])
        self.interpreter_asphalt.allocate_tensors()

        self.interpreter_desert = tf.lite.Interpreter(model_paths['desert'])
        self.interpreter_desert.allocate_tensors()

        # Get input and output details for both models
        self.input_details_asphalt = self.interpreter_asphalt.get_input_details()
        self.output_details_asphalt = self.interpreter_asphalt.get_output_details()
        self.input_details_desert = self.interpreter_desert.get_input_details()
        self.output_details_desert = self.interpreter_desert.get_output_details()

        self.timer_started = False
        self.bridge = CvBridge()
        self.current_environment = 'asphalt'
        self.teleportation_mode, self.started_teleportation = False, False
        self.pink_threshold = (np.array([145, 100, 75]), np.array([155, 255, 255]))
        self.blue_threshold = (np.array([100, 125, 40]), np.array([140, 255, 255]))

        # Dictionary with locations for teleportation
        self.board_locations = {
            'motive': {'x': -3.10, 'y': 1.50, 'z': 0.10, 'oz': -1.57, 'ow': 0.00},
            'weapon': {'x': -4.04, 'y': -2.25, 'z': 0.15, 'oz': 0.00, 'ow': 0.00},
            'bandit': {'x': -1.05, 'y': -1.20, 'z': 2.00, 'oz': 0.00, 'ow': 0.00},
        }

        # Set up ROS subscribers and publishers
        self.image_subscriber = rospy.Subscriber('/R1/pi_camera/image_raw', Image, self.image_callback)
        self.twist_publisher = rospy.Publisher('R1/cmd_vel', Twist, queue_size=1)
        self.timer_publisher = rospy.Publisher('/score_tracker', String, queue_size=1)

        rospy.sleep(1)  # Ensure subscribers and publishers are fully set up
        self.timer_publisher.publish(String('1W3B,pword,0,START'))
        self.timer_started = True

        rospy.loginfo("Driver initialized. Timer started...")

    def image_callback(self, img_msg):
        """
        Callback function for image processing, driving decisions, and managing teleportation based on the detected environment.
        """
        # Ensure timer has started before processing the image and sending commands
        if not self.timer_started:
            return

        # Ensure we are in teleportation mode before sending teleporation commands
        if self.teleportation_mode:
            if not self.started_teleportation:
                self.start_teleportation_sequence()
                self.started_teleportation = True
            return

        try:
            # Preprocess image and detect environment
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
            cv_image = cv.resize(cv_image, (128, 72))
            normalized_image = cv_image / 255.0
            hsv_image = cv.cvtColor(cv_image, cv.COLOR_BGR2HSV)
            mask_pink = cv.inRange(hsv_image, self.pink_threshold[0], self.pink_threshold[1])
            mask_pink = cv.morphologyEx(mask_pink, cv.MORPH_OPEN, np.ones((5,5), np.uint8))

            # Switch environment to desert upon detecting pink
            if cv.countNonZero(mask_pink) > (mask_pink.size * 0.01):
                self.current_environment = 'desert'
                rospy.loginfo("Pink line detected. Transitioning to desert...")

            angular_z = 0
            if self.current_environment == 'desert':
                mask_blue = cv.inRange(hsv_image, self.blue_threshold[0], self.blue_threshold[1])
                mask_blue = cv.morphologyEx(mask_blue, cv.MORPH_OPEN, np.ones((5,5), np.uint8))

                if cv.countNonZero(mask_blue) > (mask_blue.size * 0.0175):
                    self.twist_publisher.publish(Twist())
                    self.teleportation_mode = True
                    rospy.sleep(2)
                    return
                angular_z = self.predict_movement(normalized_image, 'desert')
            else:
                angular_z = self.predict_movement(normalized_image, 'asphalt')

            # Publish movement command
            self.publish_twist(angular_z)
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
        except Exception as e:
            rospy.logerr(f"Unexpected error during image callback: {e}")

    def predict_movement(self, cv_image, environment):
        """
        Predict the movement based on the current environment and the processed image.
        """
        cv_image = np.expand_dims(cv_image.astype(np.float32), axis=0)
        interpreter = self.interpreter_asphalt if environment == 'asphalt' else self.interpreter_desert
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.set_tensor(input_details[0]['index'], cv_image)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        return output_data[0][0]

    def publish_twist(self, angular_z):
        """
        Publishes the Twist message to control the robot's movement.
        """
        twist = Twist()
        twist.linear.x = 0.3645
        twist.angular.z = angular_z
        self.twist_publisher.publish(twist)

    def spawn_robot(self, x, y, z, oz, ow):
        """
        Teleport the robot to a specified location.
        """
        msg = SetModelStateRequest()
        msg.model_state.model_name = 'R1'
        msg.model_state.pose.position.x = x
        msg.model_state.pose.position.y = y
        msg.model_state.pose.position.z = z
        msg.model_state.pose.orientation.z = oz
        msg.model_state.pose.orientation.w = ow

        try:
            rospy.wait_for_service('/gazebo/set_model_state')
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            set_state(msg)
        except rospy.ServiceException as e:
            rospy.logerr(e)

    def start_teleportation_sequence(self):
        """
        Initiates the teleportation sequence to the defined board locations, ends the timer, and shuts down the node.
        """
        rospy.loginfo("Starting teleportation sequence...")
        for name, location in self.board_locations.items():
            rospy.loginfo(f"Teleporting to {name}...")
            self.spawn_robot(**location)
            rospy.sleep(2)  # Wait between teleportations for visibility
        rospy.loginfo("Teleportation sequence completed.")
        self.timer_publisher.publish(String('1W3B,password,-1,STOP'))
        rospy.loginfo("Timer ended. Shutting down the node...")

        # Shut down the ROS node
        rospy.signal_shutdown("Completed teleportation sequence and ended timer.")

    def shutdown_hook(self):
        """
        Cleans up when the ROS node is shutting down.
        """
        self.twist_publisher.publish(Twist())
        rospy.loginfo("RobotDriver shutdown.")

if __name__ == '__main__':
    rospy.init_node('robot_driver_node', anonymous=True)
    model_paths = {
        'asphalt': 'trained_model_asphalt_weight_quantized.tflite',
        'desert': 'trained_model_desert_weight_quantized.tflite'
    }
    driver = RobotDriver(model_paths)
    rospy.on_shutdown(driver.shutdown_hook)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down RobotDriver node.")
