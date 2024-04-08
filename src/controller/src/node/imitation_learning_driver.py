#!/usr/bin/env python3

import rospy
import numpy as np
import cv2 as cv
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
import tensorflow as tf
from tensorflow.keras.models import load_model

class RobotDriver:

    def __init__(self, model_paths):
        
        # # Load the TFLite model and allocate tensors
        # self.interpreter = tf.lite.Interpreter(model_path=model_path)
        # self.interpreter.allocate_tensors()

        # # Get input and output tensors
        # self.input_details = self.interpreter.get_input_details()
        # self.output_details = self.interpreter.get_output_details()

        # Load models
        self.model_asphalt = load_model(model_paths['asphalt'])
        self.model_desert = load_model(model_paths['desert'])
        self.model_offroad = load_model(model_paths['offroad'])

        self.bridge = CvBridge()
        self.current_environment = 'asphalt'
        self.pink_threshold = (np.array([145, 100, 75]), np.array([155, 255, 255]))

        # Subscribe to image_raw topic
        self.image_subscriber = rospy.Subscriber('/R1/pi_camera/image_raw', Image, self.image_callback)

        # Publish to cmd_vel topic
        self.twist_publisher = rospy.Publisher('R1/cmd_vel', Twist, queue_size=1)

        rospy.loginfo("Driver initialized.")

    def image_callback(self, img_msg):
        try:
            # Default value for angular.z
            angular_z = 0

            # General image preprocessing
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
            cv_image = cv.resize(cv_image, (128, 72))  # Resize to 128x72
            cv_image = (cv_image[int(cv_image.shape[0] * 0.4):, :])  # Crop to the bottom 60%
            normalized_image = cv_image / 255.0  # Normalize

            # Pink line detection preprocessing
            hsv_image = cv.cvtColor(cv_image, cv.COLOR_BGR2HSV)
            mask = cv.inRange(hsv_image, self.pink_threshold[0], self.pink_threshold[1])
            mask = cv.morphologyEx(mask, cv.MORPH_OPEN, np.ones((5,5), np.uint8))

            # Determine environment based on pink line detection
            self.update_environment(cv.countNonZero(mask) > (mask.size * 0.01))

            # Decide which model to use based on the current environment
            if self.current_environment in ['desert', 'transition_to_desert']:
                angular_z = self.predict_desert(normalized_image)
            elif self.current_environment in ['offroad', 'transition_to_offroad']:
                angular_z = self.predict_offroad(normalized_image)
            else:  # Default to asphalt
                angular_z = self.predict_asphalt(normalized_image)

            # Publish the angular.z component of the Twist message
            self.publish_twist(angular_z)

        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
        except Exception as e:
            rospy.logerr(f"Unexpected error during image callback: {e}")

    def update_environment(self, detected_pink_line):
        transition_map = {
            'asphalt': 'transition_to_desert',
            'desert': 'transition_to_offroad',
            'offroad': 'transition_to_desert',
            'transition_to_desert': 'desert',
            'transition_to_offroad': 'offroad'
        }
        if detected_pink_line:
            # Move to the next environment based on current
            self.current_environment = transition_map.get(self.current_environment, self.current_environment)
        else:
            # Confirm the transition if we're in one
            if 'transition' in self.current_environment:
                self.current_environment = transition_map.get(self.current_environment)

    def predict_asphalt(self, cv_image):
        # # Preprocess the image: Resize, crop, and normalize
        # cv_image = cv.resize(cv_image, (128, 72))  # Resize to 128x72
        # cv_image = cv_image[int(cv_image.shape[0] * 0.4):, :]  # Crop to the bottom 60%
        # cv_image = cv_image / 255.0  # Normalize
        
        # # Convert the image to float32 if not already
        # cv_image = cv_image.astype(np.float32)

        # # Ensure the input is in the correct shape for the TensorFlow Lite model
        # cv_image = np.expand_dims(cv_image, axis=0)  # Add batch dimension

        # # Set the input tensor
        # self.interpreter.set_tensor(self.input_details[0]['index'], cv_image)
        
        # # Run inference
        # self.interpreter.invoke()

        # # Retrieve the output from the output tensor
        # output_data = self.interpreter.get_tensor(self.output_details[0]['index'])

        # # Extract the single output value
        # angular_z = output_data[0][0]
        # return angular_z

        prediction = self.model_asphalt.predict(cv_image[None, :, :, :])
        return prediction[0][0]

    def predict_desert(self, cv_image):
        prediction = self.model_desert.predict(cv_image[None, :, :, :])
        return prediction[0][0]

    def predict_offroad(self, cv_image):
        prediction = self.model_offroad.predict(cv_image[None, :, :, :])
        return prediction[0][0]

    def publish_twist(self, angular_z):
        twist = Twist()
        twist.linear.x = 0.5  # Linear velocity
        twist.angular.z = angular_z * 2.25  # Scale output angular.z
        self.twist_publisher.publish(twist)

    def shutdown_hook(self):
        self.twist_publisher.publish(Twist())  # Stop robot
        rospy.loginfo("RobotDriver shutdown.")

if __name__ == '__main__':

    rospy.init_node('robot_driver_node', anonymous=True)
    model_paths = {
        'asphalt': 'trained_model_asphalt_2.h5',
        'desert': 'trained_model_asphalt.h5',  # train them and change them
        'offroad': 'trained_model_asphalt.h5'  # train them and change them
    }
    driver = RobotDriver(model_paths)
    rospy.on_shutdown(driver.shutdown_hook)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down RobotDriver node.")