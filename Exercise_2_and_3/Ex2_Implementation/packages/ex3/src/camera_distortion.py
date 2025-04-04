#!/usr/bin/env python3

import os
import rospy
import numpy as np
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from duckietown_msgs.msg import WheelEncoderStamped, WheelsCmdStamped, LEDPattern
from std_msgs.msg import ColorRGBA
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
        
        self._image_subcriber = rospy.Subscriber(self._camera_topic, CompressedImage, self.img_callback)
        self._image_info_subcriber = rospy.Subscriber(self._camera_info_topic, CameraInfo, self.info_callback)

        self._distorted_topic = "distorted_image"
        self._undistorted_topic = "undistorted_image"

        self._orig_publisher = rospy.Publisher(self._distorted_topic, Image, queue_size = 10)
        self._undistorted_publisher = rospy.Publisher(self._undistorted_topic, Image, queue_size = 10)

        self._camera_mtx = None
        self._proj_mtx = None
        self._d_params = None
        # From Duckiebot Portal Homography Matrix
        self.H = np.array([[-7.386314935629145e-05, 0.0008023555341693569, 0.37599498192426706],
                          [-0.002291756610528657, 2.68205480472991e-05, 0.7395497492359417],
                          [-0.00024216376950167147, 0.016003927930200265, -2.832489619617473]])
        self.H_inv = np.linalg.inv(self.H)

        self.image_new = None

        self._red_mask = None
        self._green_mask = None
        self._blue_mask = None

        # Information for wheels
        self._left_encoder_topic = f"/{self._vehicle_name}/left_wheel_encoder_node/tick" # Encoder topics for subscriber
        self._right_encoder_topic = f"/{self._vehicle_name}/right_wheel_encoder_node/tick"

        self._lane_color_dist_topic = f"/{self._vehicle_name}/line_dist_color"

        self._wheels_topic = f"/{self._vehicle_name}/wheels_driver_node/wheels_cmd" # Wheel command topic for publisher

        self._ticks_left = None
        self._ticks_right = None # Current ticks from callbacks

        self._init_ticks_left = None # Initial ticks from first callback
        self._init_ticks_right = None

        self._resolution_left = None # Number of ticks per wheel revolution 
        self._resolution_right = None

        self.line_color = None
        self.line_dist = 0.3

        # form the message for wheel publisher
        self._direction_left = 1
        self._direction_right = 1
        self._vel_left = THROTTLE * self._direction_left * 0.9
        self._vel_right = THROTTLE * self._direction_right

        # Construct publisher
        self._wheel_publisher = rospy.Publisher(self._wheels_topic, WheelsCmdStamped, queue_size=1)

        # Construct subscriber
        self.sub_left = rospy.Subscriber(self._left_encoder_topic, WheelEncoderStamped, self.callback_left)
        self.sub_right = rospy.Subscriber(self._right_encoder_topic, WheelEncoderStamped, self.callback_right)

        # Define variables for LED
        led_topic = f"/{self._vehicle_name}/led_emitter_node/led_pattern"
        self.color = ColorRGBA(255.0,255.0,255.0,255.0)

        # define other variables
        # initialize ros bag file
        self._led_publisher = rospy.Publisher(led_topic, LEDPattern, queue_size = 3)
        self._led_publisher.publish(LEDPattern(rgb_vals = [self.color,self.color,self.color,self.color,self.color]))

    def pixel_to_world(self, pixel_x, pixel_y):
        pixel = np.array([pixel_x, pixel_y, 1])  # Homogeneous coordinates
        ground = np.dot(self.H, pixel)
        ground /= ground[2]
        return ground

    def lane_dimension(self, contour):
        x, y, w, h = cv2.boundingRect(contour)  # (x, y) = top-left corner, (w, h) = width & height
        bottom_w = self.pixel_to_world(x + w/2, y + h - 1)
        distance = np.sqrt(bottom_w[0]**2 + bottom_w[1]**2)
        return distance

    def color_detect(self, image):
        hsvFrame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) 

        # Set range for red color and 
        # define mask 
        red_lower = np.array([172, 125, 50], np.uint8) 
        red_upper = np.array([179, 255, 255], np.uint8) 
        red_lower2 = np.array([0, 125, 50], np.uint8)
        red_upper2 = np.array([5, 255, 255], np.uint8)
        red_mask = cv2.inRange(hsvFrame, red_lower, red_upper) 
        red_mask2 = cv2.inRange(hsvFrame, red_lower2, red_upper2)
        red_mask = cv2.bitwise_or(red_mask, red_mask2) # to capture both higher and lower HSV values on the OpenCV reds

        # Set range for green color and 
        # define mask 
        green_lower = np.array([65, 50, 140], np.uint8)
        green_upper = np.array([95, 130, 255], np.uint8)  
        green_mask = cv2.inRange(hsvFrame, green_lower, green_upper) 

        # Set range for blue color and 
        # define mask 
        blue_lower = np.array([100, 150, 50], np.uint8)  
        blue_upper = np.array([130, 255, 255], np.uint8)
        blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper) 

        kernel = np.ones((5, 5), "uint8") 
    
        red_mask = cv2.dilate(red_mask, kernel) 
        green_mask = cv2.dilate(green_mask, kernel) 
        blue_mask = cv2.dilate(blue_mask, kernel) 

        self._red_mask = red_mask
        self._blue_mask = blue_mask
        self._green_mask = green_mask
        
        max_area = 0
        line_distance = 0

        # Creating contour to track red color 
        contours, _ = cv2.findContours(red_mask, 
                                            cv2.RETR_TREE, 
                                            cv2.CHAIN_APPROX_SIMPLE) 
        
        for contour in contours: 
            area = cv2.contourArea(contour) 
            if(area > 500 and area > max_area): 
                distance = self.lane_dimension(contour)
                line_distance = distance
                cv2.drawContours(image, contours, -1, (0, 0, 255), 2)
                 # print(f"Color RED lane distance: {distance}") 


        # Creating contour to track green color 
        contours, _ = cv2.findContours(green_mask, 
                                            cv2.RETR_TREE, 
                                            cv2.CHAIN_APPROX_SIMPLE) 
        
        for contour in contours:  
            area = cv2.contourArea(contour) 
            if(area > 500 and area > max_area): 
                distance = self.lane_dimension(contour)
                line_distance = distance
                cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
                # print(f"Color GRN lane distance: {distance}")

        # Creating contour to track blue color 
        contours, _ = cv2.findContours(blue_mask, 
                                            cv2.RETR_TREE, 
                                            cv2.CHAIN_APPROX_SIMPLE) 
        for contour in contours: 
            area = cv2.contourArea(contour) 
            if(area > 500 and area > max_area): 
                distance = self.lane_dimension(contour)
                line_distance = distance
                cv2.drawContours(image, contours, -1, (255, 0, 0), 2)
                # print(f"Color BLU lane distance: {distance}")

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
        ratio = 0.99
        height_clip = int(((height - height*ratio)/2))
        width_clip = int(((width - width*ratio)/2))

        # Crop and blur
        cropped = undistorted[height_clip:height-height_clip, width_clip:width-width_clip]
        blurred = cv2.blur(cropped, (3,3))
        
        # Encode altered image and publish
        self.image_new = blurred
        self.color_detect(blurred)

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

    def execute_blue_line_behavior(self, **kwargs):
        # Move forward while detecting correct color and over certain distance
        while (self.line_dist > 0.1):
            self.moveForward(2)
        # Stop, wait 3 sec
        stop = WheelsCmdStamped(vel_left=0, vel_right=0)
        self._wheel_publisher.publish(stop)
        time.sleep(3)

        # Blue line's behavior
        self.arcTurnRight()
        
    def execute_red_line_behavior(self, **kwargs):
        # Move forward while detecting correct color and over certain distance
        while (self.line_dist > 0.1):
            self.moveForward(2)
        # Stop, wait 3 sec
        stop = WheelsCmdStamped(vel_left=0, vel_right=0)
        self._wheel_publisher.publish(stop)
        time.sleep(3)

        # Red line's behavior
        self.moveForward(1.5)
        self.on_shutdown()
        
    def execute_green_line_behavior(self, **kwargs):
        # Move forward while detecting correct color and over certain distance
        while (self.line_dist > 0.1):
            self.moveForward(2)
        # Stop, wait 3 sec
        stop = WheelsCmdStamped(vel_left=0, vel_right=0)
        self._wheel_publisher.publish(stop)
        time.sleep(3)

        # Green line's behavior
        self.arcTurnLeft()

    def arcTurnRight(self):
        rate = rospy.Rate(20)
        turn_throttle = 0.8*THROTTLE
        self._direction_left = self._direction_right = 1

        self._vel_left = turn_throttle*self._direction_left
        self._vel_right = 0.6*turn_throttle*self._direction_right

        message = WheelsCmdStamped(vel_left=self._vel_left, vel_right=self._vel_right)

        self._init_ticks_left = self._ticks_left
        self._init_ticks_right = self._ticks_right

        signal_color = ColorRGBA(r = 255.0,g=0.0,b=0.0,a=255.0)
        self.rightSignal(self, signal_color)

        while not rospy.is_shutdown():
            if self._ticks_right is not None and self._ticks_left is not None:
                self._wheel_publisher.publish(message)
                if np.absolute(self._ticks_left - self._init_ticks_left)/self._resolution_left > (3-0.5):
                    # Stop moving
                    self._vel_left = 0
                    self._vel_right = 0
                    message = WheelsCmdStamped(vel_left=self._vel_left, vel_right=self._vel_right)
                    self._wheel_publisher.publish(message)
                    rospy.loginfo_once(f"Finished arc turn")
                    normal_color = ColorRGBA(r = 255.0,g=255.0,b=255.0,a=255.0)
                    self.oneColour(normal_color)
                    break_time = 1
                    time.sleep(break_time)
                    break
            rate.sleep()
        self.on_shutdown()

    def arcTurnLeft(self):
            rate = rospy.Rate(20)
            turn_throttle = 0.8*THROTTLE
            self._direction_left = self._direction_right = 1

            self._vel_left = 0.6*turn_throttle*self._direction_right 
            self._vel_right = turn_throttle*self._direction_left

            message = WheelsCmdStamped(vel_left=self._vel_left, vel_right=self._vel_right)

            self._init_ticks_left = self._ticks_left
            self._init_ticks_right = self._ticks_right

            signal_color = ColorRGBA(r = 255.0,g=0.0,b=0.0,a=255.0)
            self.leftSignal(self, signal_color)

            while not rospy.is_shutdown():
                if self._ticks_right is not None and self._ticks_left is not None:
                    self._wheel_publisher.publish(message)
                    if np.absolute(self._ticks_left - self._init_ticks_left)/self._resolution_left > (3-0.5):
                        # Stop moving
                        self._vel_left = 0
                        self._vel_right = 0
                        message = WheelsCmdStamped(vel_left=self._vel_left, vel_right=self._vel_right)
                        self._wheel_publisher.publish(message)
                        rospy.loginfo_once(f"Finished arc turn")
                        normal_color = ColorRGBA(r = 255.0,g=255.0,b=255.0,a=255.0)
                        self.oneColour(normal_color)
                        break_time = 1
                        time.sleep(break_time)
                        break
                rate.sleep()
            self.on_shutdown()

    def moveForward(self, desired_rotations):
            rate = rospy.Rate(20)
            self._direction_left = self._direction_right = 1
            self._vel_right = THROTTLE * self._direction_left
            self._vel_left = 0.74 * THROTTLE
            message = WheelsCmdStamped(vel_left=self._vel_left, vel_right=self._vel_right)

            self._init_ticks_left = self._ticks_left
            self._init_ticks_right = self._ticks_right

            normal_color = ColorRGBA(r = 255.0,g=255.0,b=255.0,a=255.0)
            self.oneColour(normal_color)

            while not rospy.is_shutdown():
                if self._ticks_right is not None and self._ticks_left is not None and self.line_dist is not None:
                    self._wheel_publisher.publish(message)
                    if np.absolute(self._ticks_left - self._init_ticks_left)/self._resolution_left > desired_rotations or self.line_dist < 0.1:
                        # Stop moving
                        self._vel_left = 0
                        self._vel_right = 0
                        message = WheelsCmdStamped(vel_left=self._vel_left, vel_right=self._vel_right)
                        self._wheel_publisher.publish(message)
                        rospy.loginfo_once(f"Finished moving forward")
                        break_time = 1
                        time.sleep(break_time)
                        break
                rate.sleep()

    def oneColour(self, color):
        self.color = color
        self._led_publisher.publish(LEDPattern(rgb_vals = [self.color, self.color, self.color, self.color, self.color]))

    def leftSignal(self, signal_color):
        self._led_publisher.publish(LEDPattern(rgb_vals = [signal_color,self.color,self.color,self.color,signal_color]))

    def rightSignal(self, signal_color):
        self._led_publisher.publish(LEDPattern(rgb_vals = [self.color,signal_color,signal_color,signal_color,self.color]))

    def run(self):
        # Publish image at 5 Hz
        while not rospy.is_shutdown():
            if self.image_new is not None:
                # print(f"Publishing: {self.image_new}")
                image_new = self._bridge.cv2_to_imgmsg(self.image_new, encoding="bgr8")
                self._undistorted_publisher.publish(image_new)

                # Execute the line behavior instructions for detected colour
                if self.line_color is not None:
                    if self.line_color == "BLU":
                        print("Saw colour BLU\n")
                        self.execute_blue_line_behavior()
                    elif self.line_color == "GRN":
                        print("Saw colour GRN\n")
                        self.execute_green_line_behavior()
                    elif self.line_color == "RED":
                        print("Saw colour RED\n")
                        self.execute_red_line_behavior()
            rospy.Rate(5).sleep()

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
