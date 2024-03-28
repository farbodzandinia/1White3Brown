#!/usr/bin/env python3

# Import useful packages
import rospy
import cv2 as cv
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
from std_msgs.msg import String

class ImageConverter:
    """
    A ROS node for converting images from a robot's camera into movement commands.
    It detects objects of a specific color range and controls the robot's movement.
    """

    def __init__(self):

        # Creates CvBridge instance to convert ROS image message to OpenCV format
        self.bridge = CvBridge()

        # Initializes ROS subscriber to listen for messages of type Image and call image_callback
        self.image_subscriber = rospy.Subscriber('/R1/pi_camera/image_raw', Image, self.image_callback)

        # FOR TESTING
        self.hsv_image_publisher = rospy.Publisher('/R1/hsv_image', Image, queue_size=1)
        self.binary_mask_publisher = rospy.Publisher('/R1/binary_mask', Image, queue_size=1)

        # self.clock_subscriber = rospy.Subscriber('/clock', )
        
        # Initalizes ROS publisher that sends messages of type Twist (velocity commands)
        self.velocity_publisher = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=1)

        # Initializes ROS publisher that sends the score that is of type String
        self.score_tracker_publisher = rospy.Publisher('/score_tracker', String, queue_size=1)
        
        # Settings for upper and lower thresholds in HSV for the binary mask
        self.lower_color_threshold = np.array([0, 0, 40])
        self.upper_color_threshold = np.array([180, 50, 200])
        
        # Parameters for linear velocity and an angular velocity factor (for turning)
        self.linear_velocity_setting = rospy.get_param('~linear_velocity', 0.5)
        self.angular_velocity_multiplier = rospy.get_param('~angular_velocity_factor', 0.04)

        # Confirms initialization on the terminal
        rospy.loginfo("Time_Trials initialized")

    def start_timer(self, duration):
        rospy.sleep(0.25) # Small delay to ensure everything is initialized properly
        # Publish the message to start the timer
        self.score_tracker_publisher.publish(String('1W3B,password,0,START'))
        self.timer = rospy.Timer(rospy.Duration(duration), self.stop_timer_callback, oneshot=True)
        rospy.loginfo("Timer started for {} seconds".format(duration))

    def stop_timer_callback(self, event):
        # This method will be called when the timer expires
        rospy.signal_shutdown("Timer completed")

    def stop_timer(self):
        # Send a zero velocity command to stop the robot
        self.velocity_publisher.publish(Twist())
        rospy.loginfo("Published zero velocity command")

        # Publish the message to stop the timer
        self.score_tracker_publisher.publish(String('1W3B,password,-1,STOP'))
        rospy.loginfo("Timer stopped")
        pass

    def shutdown(self):
        self.stop_timer()
        rospy.loginfo("Shutting down...")

    def image_callback(self, image_data):
        """
        Callback function for image processing.
        """

        # Tries to convert ROS image message to OpenCV format, returns error if not
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_data, "bgr8")
        except CvBridgeError as error:
            rospy.logerr(error)
            return

        # Extracts original image's height and width
        image_height, image_width = cv_image.shape[:2]

        # Crops the HSV image to the bottom 25% of the screen
        cropped_image = cv_image[int(image_height * 0.75):, :, :]

        # Converts the image from the camera from BGR to HSV format
        hsv_image = cv.cvtColor(cropped_image, cv.COLOR_BGR2HSV)
        self.hsv_image_publisher.publish(self.bridge.cv2_to_imgmsg(hsv_image, "bgr8"))

        # Applies a binary mask to the cropped image in accordance with the color thresholds
        binary_mask = cv.inRange(hsv_image, self.lower_color_threshold, self.upper_color_threshold)
        self.binary_mask_publisher.publish(self.bridge.cv2_to_imgmsg(binary_mask, "mono8"))
        
        # Finds extreme external contours in the binary image
        contours, _ = cv.findContours(binary_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # Loops through each sizable contour
        for contour in [c for c in contours if cv.contourArea(c) > 1]:
            
            # Approximates the size of the smallest rectangle within the contour
            x, _, w, _ = cv.boundingRect(contour)

            # Calculates the midpoint of the lane using the top-left coordinate and the width of the rectangle
            lane_center_x = int(x + w / 2)

            # Calculates angular velocity based on the position of the lane center with respect to the camera
            angular_velocity = self.angular_velocity_multiplier * (image_width / 2 - lane_center_x)

            # Updates velocities and publishes the Twist message to the robot
            movement_command = Twist()
            movement_command.linear.x = self.linear_velocity_setting
            movement_command.angular.z = angular_velocity
            self.velocity_publisher.publish(movement_command)

def main():

    try:
        rospy.init_node('image_converter', anonymous=True)
        ic = ImageConverter()
        ic.start_timer(5)
        rospy.on_shutdown(ic.shutdown)  # Register the shutdown hook
        rospy.spin()

    except KeyboardInterrupt:
        # This block allows the node to be shut down with Ctrl+C from the command line.
        rospy.signal_shutdown('KeyboardInterrupt')
        raise

if __name__ == '__main__':
    main()