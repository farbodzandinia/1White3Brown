#!/usr/bin/env python3

import rospy
import cv2 as cv
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
import tensorflow as tf
from tensorflow.keras.models import load_model

class RobotDriver:

    def __init__(self, model):
        
        self.model = model
        self.bridge = CvBridge()

        # Subscribe to image_raw topic
        self.image_subscriber = rospy.Subscriber('/R1/pi_camera/image_raw', Image, self.image_callback)
        # Publish to cmd_vel topic
        self.twist_publisher = rospy.Publisher('R1/cmd_vel', Twist, queue_size=1)

        # Initialize the ROS node
        
        rospy.loginfo("Driver initialized.")

    def image_callback(self, img_msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
            cv_image = cv.resize(cv_image, (128, 72))  # Resize to 128x72
            cv_image = cv_image[int(cv_image.shape[0] * 0.4):, :]  # Crop to the bottom 60%
            cv_image = cv_image / 255  # Normalize
            angular_z = self.predict(cv_image)
            self.publish_twist(angular_z)
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
        except Exception as e:
            rospy.logerr(f"Unexpected error during image callback: {e}")

    def predict(self, cv_image):
        prediction = self.model.predict(cv_image[None, :, :, :])
        angular_z = prediction[0][0]  # Model outputs a single value
        return angular_z

    def publish_twist(self, angular_z):
        twist = Twist()
        twist.linear.x = 0.5  # Linear velocity
        twist.angular.z = angular_z * 2  # Scale output angular.z
        self.twist_publisher.publish(twist)

    def shutdown_hook(self):
        self.twist_publisher.publish(Twist())  # Stop robot
        rospy.loginfo("RobotDriver shutdown.")

if __name__ == '__main__':

    rospy.init_node('robot_driver_node', anonymous=True)
    model_name = 'trained_model.keras'

    model = load_model(model_name)
    driver = RobotDriver(model)
    
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    
    rospy.on_shutdown(driver.shutdown_hook)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down RobotDriver node.")