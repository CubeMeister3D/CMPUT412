#!/usr/bin/env python3

import os
import rospy
import numpy as np
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from duckietown_msgs.msg import WheelEncoderStamped, WheelsCmdStamped
import time
import cv2
from cv_bridge import CvBridge

THROTTLE = 0.3

class CameraReadWriteNode(DTROS):
    def __init__(self, node_name):
        # initialize the DTROS parent class
        super(CameraReadWriteNode, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)

        self._vehicle_name = os.environ['VEHICLE_NAME']
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        self._camera_info_topic = f"/{self._vehicle_name}/camera_node/camera_info"

        self._bridge = CvBridge()
        
        self._image_subcriber = rospy.Subscriber(self._camera_topic, CompressedImage, self.img_callback, queue_size=1)
        self._image_info_subcriber = rospy.Subscriber(self._camera_info_topic, CameraInfo, self.info_callback)

        self._undistorted_topic = "undistorted_image"
        self._undistorted_publisher = rospy.Publisher(self._undistorted_topic, Image, queue_size=1)

        self._camera_mtx = None
        self._proj_mtx = None
        self._d_params = None
        # From Duckiebot Portal Homography Matrix
        self.H = np.array([[-7.386314935629145e-05, 0.0008023555341693569, 0.37599498192426706],
                          [-0.002291756610528657, 2.68205480472991e-05, 0.7395497492359417],
                          [-0.00024216376950167147, 0.016003927930200265, -2.832489619617473]])
        self.H_inv = np.linalg.inv(self.H)

        self.image_new = None

        # Information for wheels
        self._left_encoder_topic = f"/{self._vehicle_name}/left_wheel_encoder_node/tick" # Encoder topics for subscriber
        self._right_encoder_topic = f"/{self._vehicle_name}/right_wheel_encoder_node/tick"

        self._wheels_topic = f"/{self._vehicle_name}/wheels_driver_node/wheels_cmd" # Wheel command topic for publisher

        self._ticks_left = None
        self._ticks_right = None # Current ticks from callbacks

        self._init_ticks_left = None # Initial ticks from first callback
        self._init_ticks_right = None

        self._resolution_left = None # Number of ticks per wheel revolution 
        self._resolution_right = None

        # Node states
        self._can_see_blue = False
        self._can_see_ducks = False
        self.line_area = 0
        self.state = "IDLE" # States: IDLE, MOVING, STOPPED
        self.has_stopped = False
        self.prev_time = time.time()

        # form the message for wheel publisher
        self._direction_left = 1
        self._direction_right = 1
        self._vel_left = THROTTLE * self._direction_left * 0.5
        self._vel_right = THROTTLE * self._direction_right

        # Construct publisher
        self._wheel_publisher = rospy.Publisher(self._wheels_topic, WheelsCmdStamped, queue_size=1)

        # Construct subscriber
        self.sub_left = rospy.Subscriber(self._left_encoder_topic, WheelEncoderStamped, self.callback_left)
        self.sub_right = rospy.Subscriber(self._right_encoder_topic, WheelEncoderStamped, self.callback_right)

    def color_detect(self, image):
        # Change image to HSV color space
        if image is not None:
            hsvFrame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) 

            # Set range for blue color and define mask 
            blue_lower = np.array([100, 150, 50], np.uint8)  
            blue_upper = np.array([130, 255, 255], np.uint8)
            blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper) 

            # Set range for yellow color and define mask 
            yellow_lower = np.array([14, 100, 175], np.uint8)  
            yellow_upper = np.array([19, 210, 255], np.uint8)
            yellow_mask = cv2.inRange(hsvFrame, yellow_lower, yellow_upper) 

            # Brush/dilate with kernel to reduce noise in mask
            kernel = np.ones((5, 5), "uint8") 
            blue_mask = cv2.dilate(blue_mask, kernel) 
            yellow_mask = cv2.dilate(yellow_mask, kernel) 
            
            # Creating contour to track blue color (find largest)
            max_area = 0
            x = 0
            y = 0
            w = 0
            h = 0
            contours, _ = cv2.findContours(blue_mask, 
                                                cv2.RETR_TREE, 
                                                cv2.CHAIN_APPROX_SIMPLE) 
            for contour in contours: 
                area = cv2.contourArea(contour) 
                if (area > 500 and area > max_area):
                    max_area = area 
                    x, y, w, h = cv2.boundingRect(contour)
            
            # Draw bounding box around largest blue contour (the blue line)
            if max_area > 0:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            self.line_area = max_area

            self._can_see_ducks = False
            # Creating contour to track yellow color 
            contours, _ = cv2.findContours(yellow_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    self._can_see_ducks = True # Set state to True if yellow contour found is large enough
            
            self.image_new = image

    def img_callback(self, msg):
        # convert JPEG bytes to CV image
        image = self._bridge.compressed_imgmsg_to_cv2(msg)
        height, width, _ = image.shape
        # print(f"Shape {image.shape} - Format: {image.dtype}")
        undistorted = image

        # Undistort image
        if self._camera_mtx is not None and self._proj_mtx is not None and self._d_params is not None:
            undistorted = cv2.undistort(image, self._camera_mtx, self._d_params)

        # Set ratio (% size of original image) for cropping and amt to clip from all sides
        ratio = 0.4
        height_clip = int(height * (1 - ratio))
        width_clip = int(((width - width*0.80)/2))

        # Crop and blur
        cropped = undistorted[height_clip:height, width_clip:width - width_clip]
        blurred = cv2.blur(cropped, (3, 3))
        
        # Encode altered image and publish
        self.image_new = blurred
        self.color_detect(blurred)
        image_new = self._bridge.cv2_to_imgmsg(self.image_new, encoding="bgr8")
        self._undistorted_publisher.publish(image_new)

    def callback_left(self, data):
        # log general information once at the beginning
        rospy.loginfo_once(f"Left encoder resolution: {data.resolution}")
        rospy.loginfo_once(f"Left encoder type: {data.type}")
        # store data value
        self._ticks_left = data.data

        # If INITIAL ticks not set, store them once
        if not self._init_ticks_left:
            self._init_ticks_left = data.data
            self._resolution_left = data.resolution

    def callback_right(self, data):
        # log general information once at the beginning
        rospy.loginfo_once(f"Right encoder resolution: {data.resolution}")
        rospy.loginfo_once(f"Right encoder type: {data.type}")
        # store data value
        self._ticks_right = data.data

        # If INITIAL ticks not set, store them once
        if not self._init_ticks_right:
            self._init_ticks_right = data.data
            self._resolution_right = data.resolution

    def stopMovement(self):
        stop = WheelsCmdStamped(vel_left=0, vel_right=0)
        self._wheel_publisher.publish(stop)
        self.state = "STOPPED"
        self.has_stopped = True

    def moveForward(self, desired_rotations):
            # If idle and send to moveForward, reset movement
            message = WheelsCmdStamped(vel_left=self._vel_left, vel_right=self._vel_right)

            if self.state == "IDLE":
                self._direction_left = self._direction_right = 1
                self._vel_right = THROTTLE * self._direction_left
                self._vel_left = 0.6 * THROTTLE
                self._init_ticks_left = self._ticks_left
                self._init_ticks_right = self._ticks_right

                self.state = "MOVING"

            # If we have tick information for stepper motors + the bot is in the MOVING STATE, then we can move forward
            if self._ticks_right is not None and self._ticks_left is not None and self.state == "MOVING":
                self._wheel_publisher.publish(message)

                # Stop moving if we've reached the desired length OR we've got a valid (Blue) line distance < 10cm
                if np.absolute(self._ticks_left - self._init_ticks_left)/self._resolution_left > desired_rotations or (self.line_area is not None and self.line_area > 7000) and not self.has_stopped:
                    if (self.line_area > 7000):
                        rospy.loginfo(f"Stopped moving forward - BLUE LINE")
                    else: 
                        rospy.loginfo(f"Stopped moving forward - End of LINE")
                    # Call stop moving method, change state
                    self.stopMovement()

    def run(self):
        # Publish image, commands at 5 Hz
        while not rospy.is_shutdown():
            if self.image_new is not None:
                # Move forward; control returns to run IF finished or encountered nearby blue line
                rospy.loginfo(f"Running... | Ducks: {self._can_see_ducks}, Line: {self.line_area}, State: {self.state}")
                if self.line_area == 0:
                    self.has_stopped = False # Reset has_stopped whenever line not visible to Duckiebot

                if self.state == "IDLE" and not self._can_see_ducks and time.time() - self.prev_time >= 1:
                    rospy.loginfo("Moving...")
                    self.moveForward(6)      
                
                elif self.state == "MOVING":
                    if self.line_area > 7000 and not self.has_stopped: # If stopped, then ignore (for a bit)
                        self.stopMovement()
                        rospy.loginfo(f"Not moving - blue line visible.")
                        self.prev_time = time.time()
                    elif self._can_see_ducks and time.time() - self.prev_time >= 1: # Enforce 1 sec waiting without blocking
                        rospy.loginfo("Not moving - ducks visible ")
                    else:
                        self.state = "IDLE"
                        self.moveForward(6)
                
                elif self.state == "STOPPED" and time.time() - self.prev_time >= 1: # Enforce 1 sec waiting without blocking
                    if self._can_see_ducks:
                        rospy.loginfo("Not moving - ducks visible ")
                    else:
                        self.state = "IDLE" 
                        self.moveForward(6)   
            rospy.Rate(20).sleep()

    def on_shutdown(self):
        stop = WheelsCmdStamped(vel_left=0, vel_right=0)
        self._wheel_publisher.publish(stop)

    def info_callback(self, msg):
        if self._camera_mtx is None:
            message = msg.K
            self._camera_mtx =  np.array([(message[0:3]), (message[3:6]), (message[6:9])], dtype='float32')
            rospy.loginfo_once(f"Camera Matrix: {self._camera_mtx}")
        if self._proj_mtx is None:
            message = msg.P
            self._proj_mtx =  np.array([(message[0:3]), (message[3:6]), (message[6:9])], dtype='float32')
            rospy.loginfo_once(f"Projection Matrix: {self._proj_mtx}")
        if self._d_params is None:
            self._d_params = np.array(msg.D)
            rospy.loginfo_once(f"Distortion Parameters: {self._d_params}")


if __name__ == '__main__':
    # create the node
    node = CameraReadWriteNode(node_name='camera_read_write_node')
    node.run()
    # keep spinning
    rospy.spin()
