#!/usr/bin/env python3

import rospy
import cv2 as cv
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
from tensorflow.keras.models import load_model

class RobotDriver:

    def __init__(self, model_path):
        self.model = load_model(model_path)
        self.bridge = CvBridge()

        # Subscribe to image_raw topic
        self.image_subscriber = rospy.Subscriber('/R1/pi_camera/image_raw', Image, self.image_callback)
        # Publish to cmd_vel topic
        self.twist_publisher = rospy.Publisher('R1/cmd_vel', Twist, queue_size=1)

        # Initialize the ROS node
        rospy.init_node('robot_driver_node', anonymous=True)
        rospy.loginfo("Driver initialized.")

    def image_callback(self, img_msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
            cv_image = cv.resize(cv_image, (128, 72))  # Resize to 128x72
            cv_image = cv_image[int(cv_image.shape[0] * 0.4):, :]  # Crop to the bottom 60%
            angular_z = self.predict(cv_image)
            self.publish_twist(angular_z)
        except CvBridgeError as e:
            rospy.logerr(e)

    def predict(self, cv_image):
        prediction = self.model.predict(cv_image[None, :, :, :])
        angular_z = prediction[0][0]  # Model outputs a single value
        return angular_z

    def publish_twist(self, angular_z):
        twist = Twist()
        twist.linear.x = 0.5  # Linear velocity
        twist.angular.z = angular_z  # Scale if you need to
        self.twist_publisher.publish(twist)

    def shutdown_hook(self):
        rospy.loginfo("RobotDriver shutdown.")

if __name__ == '__main__':

    model_path = 'trained_model.keras'
    driver = RobotDriver(model_path)
    rospy.on_shutdown(driver.shutdown_hook)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down RobotDriver node.")