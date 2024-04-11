# !/usr/bin/env python3

import cv2
import numpy as np
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import string
import tensorflow as tf

# Define the ClueboardDetector class
class ClueboardDetector:
    def __init__(self):
        self.interpreter = tf.lite.Interpreter(model_path='quantized_model.tflite')
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.bridge = CvBridge()

        rospy.Subscriber('/R1/pi_camera/image_raw', Image, callback)
        self.score_publisher = rospy.Publisher('/score_tracker', String, queue_size=1)
        rospy.Subscriber('/score_tracker', String, self.score_callback)

        self.characters = " " + string.ascii_uppercase + string.digits
        self.label_dict = {char: i for i, char in enumerate(self.characters)}
        self.index_to_char = {index: char for char, index in self.label_dict.items()}

    def detect_clueboard(self, cv_image):
        # Convert to HSV color space for easier color thresholding
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # Based on the image, these bounds are adjusted for a darker blue
        blue = (np.array([100, 125, 40]), np.array([140, 255, 255]))  # Thresholds for dark blue

        # Create a mask with the new bounds
        mask = cv2.inRange(hsv, blue[0], blue[1])

        # Find contours in the edge-detected image
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Find the outermost contour
        if len(contours) > 0:
            # Sort the contours by area in descending order (largest first)
            sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
            if len(sorted_contours) > 1:
                needed_contour = sorted_contours[1]
                epsilon = 0.05 * cv2.arcLength(needed_contour, True)
                approx = np.squeeze(cv2.approxPolyDP(needed_contour, epsilon, True))
                approx = order_points(approx)

                if len(approx) == 4:  # Ensure the contour has 4 points
                    # Assuming the detected contour approximates the corners of the signboard
                    pts1 = np.float32([approx[0], approx[1], approx[2], approx[3]])
                    # Define points for the desired output (signboard dimensions)
                    signboard_width, signboard_height = 600, 400
                    pts2 = np.float32([[0, 0], [signboard_width, 0], [signboard_width, signboard_height], [0, signboard_height]])
                    # Calculate the perspective transform matrix and apply it
                    matrix = cv2.getPerspectiveTransform(pts1, pts2)
                    signboard_transformed = cv2.warpPerspective(cv_image, matrix, (signboard_width, signboard_height))
                    return signboard_transformed
                else:
                    return None 
        return None  # Return None if no signboard is detected

    def shutdown_hook(self):
        rospy.loginfo("ClueboardDetector shutdown.")

    # Define a function to preprocess and extract letters from the detected signboard
    def preprocess_and_extract_letters(self, signboard_image):
        # Convert to grayscale and apply denoising and thresholding in one step
        image = cv2.cvtColor(signboard_image, cv2.COLOR_BGR2GRAY)
        image = cv2.fastNlMeansDenoising(image, None, 30, 7, 21)
        image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # Enhance edges using Laplacian and combine
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        signboard_image = cv2.addWeighted(image, 1.5, np.uint8(np.absolute(laplacian)), -0.5, 0)

        # Assuming image is already in grayscale so the condition check is removed
        h, w = signboard_image.shape

        # Pre-calculate crop percentages as ratios to improve readability
        crop_ratios = {
            'category_top': 0.05, 'category_bottom': 0.65, 'category_left': 0.42, 'category_right': 0.13,
            'word_top': 0.55, 'word_bottom': 0.15, 'word_left': 0.05, 'word_right': 0.05
        }

        # Apply cropping based on pre-calculated ratios
        cropped_image_category = signboard_image[int(h * crop_ratios['category_top']):int(h * (1 - crop_ratios['category_bottom'])),
                                                int(w * crop_ratios['category_left']):int(w * (1 - crop_ratios['category_right']))]
        cropped_image_word = signboard_image[int(h * crop_ratios['word_top']):int(h * (1 - crop_ratios['word_bottom'])),
                                            int(w * crop_ratios['word_left']):int(w * (1 - crop_ratios['word_right']))]

        # Simplified letter extraction using comprehension
        def extract_letters(cropped_image, n_letters):
            letter_width = cropped_image.shape[1] // n_letters
            return [cv2.resize(cropped_image[:, i*letter_width:(i+1)*letter_width], (45, 120)).astype(np.float32)[np.newaxis, ...]
                    for i in range(n_letters)]

        # Use the simplified function to extract and preprocess letters for 'category' and 'word'
        preprocessed_letters_category = np.array(extract_letters(cropped_image_category, 6))
        preprocessed_letters_word = np.array(extract_letters(cropped_image_word, 12))

        return preprocessed_letters_category, preprocessed_letters_word

    def run_tflite_model(self, input_data):
        input_data = np.expand_dims(input_data, axis=-1)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        return output_data

    def score_callback(self, data):
        # Check if the received data is "-1", indicating a shutdown signal
        if data.data == "-1":
            rospy.signal_shutdown("Received shutdown signal from score tracker.")

# Callback function for the image subscriber
def callback(ros_image):
    try:
        # Convert the ROS image to an OpenCV image
        cv_image = detect.bridge.imgmsg_to_cv2(ros_image, desired_encoding = "bgr8")
    except CvBridgeError as e:
        rospy.logerr(e)
        return

    # Instantiate the ClueboardDetector and detect the signboard in the image
    signboard_image = detect.detect_clueboard(cv_image)

    if signboard_image is not None:
        #Preprocess the signboard image and extract letters
        preprocessed_letters_category, preprocessed_letters_word = detect.preprocess_and_extract_letters(signboard_image)

        # Use the CNN model to predict letters
        predicted_labels_category = []
        predicted_labels_word = []

        for letter in preprocessed_letters_category:
            letter = detect.run_tflite_model(letter)
            predicted_labels_category.append(letter)

        for letter in preprocessed_letters_word:
            letter = detect.run_tflite_model(letter)
            predicted_labels_word.append(letter)

        # Assuming predicted_labels_category and predicted_labels_word contain model predictions
        predicted_indices_category = [np.argmax(letter) for letter in predicted_labels_category]
        predicted_indices_word = [np.argmax(letter) for letter in predicted_labels_word]

        # Decode the predictions to characters
        decoded_category = ''.join(detect.index_to_char[index] for index in predicted_indices_category)

        decoded_word = ''.join(detect.index_to_char[index] for index in predicted_indices_word)
        
        categories = ["SIZE", "VICTIM", "CRIME", "TIME", "PLACE", "MOTIVE", "WEAPON", "BANDIT"]

        # Initialize a variable to store the found index
        location = None

        # Use a for loop with enumerate to find the index
        for index, string in enumerate(categories):
            if string[0] == decoded_category[0]:
                if string[1] == decoded_category[1]:
                    location = index+1
                    break  # Exit the loop once the string is found

        # Construct the message with team information and prediction to send to the score tracker
        if location is not None:
            team_id = "1W3B"
            team_password = "pword"
            clue_location = location
            clue_prediction = decoded_word
            message_data = f"{team_id},{team_password},{clue_location},{clue_prediction}"
            message = String(data=message_data)
            detect.score_publisher.publish(message)

def order_points(pts):
    if len(pts.shape) >= 2 and pts.shape[1] == 2:

        #Calculate the sum and difference of the points' coordinates
        sum_pts = pts.sum(axis=1)
        diff_pts = np.diff(pts, axis=1)

        # The bottom-left point will have the smallest sum, whereas
        # the top-right point will have the largest sum
        top_left = pts[np.argmin(sum_pts)]
        bottom_right = pts[np.argmax(sum_pts)]

        # The top-left point will have the smallest difference,
        # whereas the bottom-right point will have the largest difference
        top_right = pts[np.argmin(diff_pts)]
        bottom_left = pts[np.argmax(diff_pts)]

        # Return the coordinates in the order: top-left, top-right, bottom-right, bottom-left
        return np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")

if __name__ == '__main__':

    # Initialize the ROS node
    rospy.init_node('clue_detection', anonymous=True)
    detect = ClueboardDetector()

    rospy.on_shutdown(detect.shutdown_hook)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down ClueboardDetector node.")