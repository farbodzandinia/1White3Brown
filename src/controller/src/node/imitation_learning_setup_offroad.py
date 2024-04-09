#!/usr/bin/env python3

import os
import csv
import rospy
import datetime
import cv2
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError

class DataCollectorOffroad:

    def __init__(self, save_directory):
        self.bridge = CvBridge()
        self.image_subscriber = rospy.Subscriber('/R1/pi_camera/image_raw', Image, self.image_callback)
        self.twist_subscriber = rospy.Subscriber('R1/cmd_vel', Twist, self.twist_callback)
        self.last_cmd = Twist()
        self.save_directory = save_directory
        self.img_count = 0
        self.last_saved_time = rospy.Time.now()
        self.save_interval = rospy.Duration(0.001)  # Screenshot every 0.01s

        if not os.path.exists(self.save_directory):
            os.makedirs(self.save_directory)

        # Check if the CSV file already exists
        csv_file_path = os.path.join(self.save_directory, "data.csv")
        file_exists = os.path.isfile(csv_file_path)

        # Open the file in append mode, so new data is added to the end without erasing existing data
        self.csv_file = open(csv_file_path, "a")
        self.csv_writer = csv.writer(self.csv_file)

        # If the file doesn't exist, write the header
        if not file_exists:
            self.csv_writer.writerow(['image_name', 'angular_z', 'linear_x'])

        rospy.loginfo("DataCollectorOffroad initialized.")

    def image_callback(self, img_msg):

        current_time = rospy.Time.now()
        if current_time - self.last_saved_time >= self.save_interval:
            # Update the time of the last saved image
            self.last_saved_time = current_time

            # Process and resize the image
            try:
                cv_image = cv2.resize(self.bridge.imgmsg_to_cv2(img_msg, "bgr8"), (128, 72))  # Resize to 128x72 (16:9)
                cv_image = cv_image[int(cv_image.shape[0] * (0.4)):, :]  # Then crop to the bottom 60%
            except CvBridgeError as error:
                rospy.logerr(error)
                return

            # Figure out the timestamp for naming the image
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S_%f")
            image_filename = "image_{}.jpeg".format(timestamp)
            image_path = os.path.join(self.save_directory, image_filename)

            # Save the image
            cv2.imwrite(image_path, cv_image)

            # Log the data (file name, angular.z, linear.x)
            self.csv_writer.writerow([image_filename, self.last_cmd.angular.z, self.last_cmd.linear.x])
            self.img_count += 1
            rospy.loginfo("Saved image #{}".format(self.img_count))

    def twist_callback(self, cmd_msg):
        self.last_cmd = cmd_msg

    def shutdown_hook(self):
        self.csv_file.close()
        rospy.loginfo("DataCollectorOffroad shutdown, saved {} images.".format(self.img_count))

if __name__ == "__main__":
    
    rospy.init_node('data_collector_offroad_node', anonymous=True)
    save_dir = rospy.get_param('~save_dir', 'data_offroad')
    collector = DataCollectorOffroad(save_dir)
    rospy.on_shutdown(collector.shutdown_hook)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down DataCollectorOffroad node.")
