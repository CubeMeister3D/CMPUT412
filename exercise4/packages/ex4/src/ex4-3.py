#!/usr/bin/env python3

import os
import rospy
import numpy as np
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage, Image
from sensor_msgs.msg import CameraInfo
from duckietown_msgs.msg import Twist2DStamped, WheelEncoderStamped, WheelsCmdStamped, LEDPattern


import cv2
from cv_bridge import CvBridge

import time

CONTROLLER = "PID"

THROTTLE = 0.4

class LaneBehaviourNode(DTROS):
    def __init__(self, node_name):
        # initialize the DTROS parent class
        super(LaneBehaviourNode, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)

        self._vehicle_name = os.environ['VEHICLE_NAME']
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        self._camera_info_topic = f"/{self._vehicle_name}/camera_node/camera_info"

        self.last_process_time = None

        self._bridge = CvBridge()
        
        self._image_subcriber = rospy.Subscriber(self._camera_topic, CompressedImage, self.img_callback)
        self._image_info_subcriber = rospy.Subscriber(self._camera_info_topic, CameraInfo, self.info_callback)

        self._contour_img_topic = "contour_img"
        self._contour_img_publisher = rospy.Publisher(self._contour_img_topic, Image, queue_size = 10)

        self._camera_mtx = None
        self._proj_mtx = None
        self._d_params = None
        # From Duckiebot Portal Homography Matrix
        self.H = np.array([[-7.386314935629145e-05, 0.0008023555341693569, 0.37599498192426706],
                          [-0.002291756610528657, 2.68205480472991e-05, 0.7395497492359417],
                          [-0.00024216376950167147, 0.016003927930200265, -2.832489619617473]])
        self.H_inv = np.linalg.inv(self.H)

        self.image_new = None

        self._wht_mask = None
        self._ylw_mask = None

        # For P, D, I terms
        self.error = 0
        self.prev_error = 0
        self.sum_of_error = 0

        self._yellow_distance = 0.5
        self._white_distance = 0.5


        # WHEEL/CONTROL TOPICS

        # form the message for wheel publisher
        self.gain = 0.02
        self.d_gain = 0.01
        self.i_gain = 0.01
        self._linear_velo = 0.15

        twist_topic = f"/{self._vehicle_name}/car_cmd_switch_node/cmd"
        self.movement_publisher = rospy.Publisher(twist_topic, Twist2DStamped, queue_size=1)

        # States
        self.bot_area = 0 
        self.state = "MOVING" # States: MOVING (Under PD/PID controller control), STOPPED (Assessing 3 sec), OVERRIDE (Under Dead-Reckoning Control)
        self.prev_time = time.time()

        self._left_encoder_topic = f"/{self._vehicle_name}/left_wheel_encoder_node/tick" # Encoder topics for subscriber
        self._right_encoder_topic = f"/{self._vehicle_name}/right_wheel_encoder_node/tick"

        self._wheels_topic = f"/{self._vehicle_name}/wheels_driver_node/wheels_cmd" # Wheel command topic for publisher

        self._ticks_left = None
        self._ticks_right = None # Current ticks from callbacks

        self._init_ticks_left = None # Initial ticks from first callback
        self._init_ticks_right = None

        self._resolution_left = None # Number of ticks per wheel revolution 
        self._resolution_right = None

        # form the message for wheel publisher
        self._direction_left = 1
        self._direction_right = 1
        self._vel_left = THROTTLE * self._direction_left * 0.7
        self._vel_right = THROTTLE * self._direction_right

        # Construct publisher
        self._publisher = rospy.Publisher(self._wheels_topic, WheelsCmdStamped, queue_size=1)

        # Construct subscriber
        self.sub_left = rospy.Subscriber(self._left_encoder_topic, WheelEncoderStamped, self.callback_left)
        self.sub_right = rospy.Subscriber(self._right_encoder_topic, WheelEncoderStamped, self.callback_right)
    
    def find_centroid(self, contour):
        M_cont = cv2.moments(contour) # Find 'moments' of contour
        if M_cont["m00"] != 0:
            # If area of the contour is not 0, then the x-val of centroid = all x coords / area
            return int(M_cont["m10"] / M_cont["m00"])  # Centroid x value
        else:
            return None # Failed to find centroid

    def find_error(self, contour_yellow, contour_white):
        # Find the 'error' of the centerline; avg((distance of yellow from center) + (distance of white from center)) - real center
        # AKA: estimated center using lines - real center of image
        _, width, _ = self.image_new.shape
        center_x = width // 2 # Center of image wrt. x

        # Find the central x value inside the contours
        cx_yellow = self.find_centroid(contour_yellow)
        cx_white = self.find_centroid(contour_white)

        if cx_yellow is not None and cx_white is not None:
            # Return estimated centerline minus real centerline wrt. image frame
            return (cx_white+cx_yellow)//2 - center_x
        else:
            # If one or more centroids DNE, then return None as operation failed.
            return None

    def color_detect(self, image):
        hsvFrame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) # Change image to HSV color space

        # Set range for blue color and define mask 
        blue_lower = np.array([102, 150, 50], np.uint8)  
        blue_upper = np.array([109, 255, 120], np.uint8)
        blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper) 

        # Set range for white color and define mask 
        white_lower = np.array([117, 0, 140], np.uint8)
        white_upper = np.array([133, 40, 255], np.uint8)  
        white_mask = cv2.inRange(hsvFrame, white_lower, white_upper) 

        # Set range for yellow color and define mask 
        yellow_lower = np.array([23, 120, 110], np.uint8)  
        yellow_upper = np.array([29, 255, 255], np.uint8)
        yellow_mask = cv2.inRange(hsvFrame, yellow_lower, yellow_upper) 

        # Reduce noise from masks by brushing using kernel
        kernel = np.ones((5, 5), "uint8") 
        white_mask = cv2.dilate(white_mask, kernel) 
        yellow_mask = cv2.dilate(yellow_mask, kernel) 

        # Update Node mask attributes
        self._wht_mask = white_mask
        self._ylw_mask = yellow_mask
        
        max_yellow_area = 0
        max_white_area = 0

        yellow_contour = None
        white_contour = None

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
        self.bot_area = max_area

        # Creating contour to track yellow color 
        contours, _ = cv2.findContours(yellow_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Find the largest yellow contour, save
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500 and area > max_yellow_area:
                yellow_contour = contour
                max_yellow_area = area  # Update the max yellow area
                x, y, w, h = cv2.boundingRect(contour)

        # Draw the bounding box for the largest yellow contour 
        if yellow_contour is not None:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Creating contour to track white color 
        contours, _ = cv2.findContours(white_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Find the largest white contour, save
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500 and area > max_white_area:
                white_contour = contour
                max_white_area = area  # Update the max white area
                x, y, w, h = cv2.boundingRect(contour)

        # Draw the bounding box for the largest white contour 
        if white_contour is not None:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), 2)
        
        # Update Node attributes
        self.image_new = image

        # Finding error, yellow, white distances
        _, width, _ = self.image_new.shape
        center_x = width // 2

        # Default values for error, yellow, white distances
        self.error = None
        self._yellow_distance = None
        self._white_distance = None

        if yellow_contour is not None and white_contour is not None:
            # If BOTH lines visible, error can be found
            error = self.find_error(yellow_contour, white_contour)
            self.error = error
            self._yellow_distance = (self.find_centroid(yellow_contour) - center_x)//2
            self._white_distance = (self.find_centroid(white_contour) - center_x)//2
        elif yellow_contour is not None:
            # If only yellow visible, then find yellow distance from center / 2
            self._yellow_distance = (center_x - self.find_centroid(yellow_contour))
        elif white_contour is not None:
            # If only white visible, then find white distance from center / 2
            self._white_distance = (center_x - self.find_centroid(white_contour))

        # Publish contour outline image on callback + process
        image_new = self._bridge.cv2_to_imgmsg(self.image_new, encoding="bgr8")
        self._contour_img_publisher.publish(image_new)

    def img_callback(self, msg):
        # convert JPEG bytes to CV image
        current_time = time.time()
        if self.last_process_time is None or current_time - self.last_process_time >= 0.1:
            self.last_process_time = current_time 
        else:
            return 
        
        image = self._bridge.compressed_imgmsg_to_cv2(msg)
        height, width, _ = image.shape
        # print(f"Shape {image.shape} - Format: {image.dtype}")
        undistorted = image

        # Undistort image
        if self._camera_mtx is not None and self._proj_mtx is not None and self._d_params is not None:
            undistorted = cv2.undistort(image, self._camera_mtx, self._d_params)

        # Set ratio (% size of original image) for cropping and amt to clip from all sides
        ratio = 0.45
        height_clip = int(height * (1 - ratio))
        width_clip = int(((width - width*0.80)/2))

        # Crop and blur
        cropped = undistorted[height_clip:height, width_clip:width - width_clip]
        blurred = cv2.blur(cropped, (3,3))
        
        # Color detection w altered image
        self.color_detect(blurred)

    def info_callback(self, msg):
        # Save camera_info to the Node
        if self._camera_mtx is None: # Camera matrix
            message = msg.K
            self._camera_mtx =  np.array([(message[0:3]), (message[3:6]), (message[6:9])], dtype='float32')
            rospy.loginfo_once(f"Camera Matrix: {self._camera_mtx}")
        if self._proj_mtx is None: # Projection matrix
            message = msg.P
            self._proj_mtx =  np.array([(message[0:3]), (message[3:6]), (message[6:9])], dtype='float32')
            rospy.loginfo_once(f"Projection Matrix: {self._proj_mtx}")
        if self._d_params is None: # Distortion parameters (INTRINSICS)
            self._d_params = np.array(msg.D)
            rospy.loginfo_once(f"Distortion Parameters: {self._d_params}")

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

    def calculate_p_control(self, **kwargs):
        # Proportional Controller
        if self.error is not None: # If error exists, then use w = K*error 
            return self.gain * self.error
        elif self._white_distance is not None: # If error does not exist, look for white line: w = K*white_error
            return self.gain * self._white_distance
        elif self._yellow_distance is not None: # If only yellow line visible: w = K*yellow_error
            return self.gain * self._yellow_distance 
        self._angular_velo = 0

    def calculate_pd_control(self, **kwargs):
        # Proportional Derivative
        current_error = 0
        if self.error is not None:
            current_error = self.error
        elif self._white_distance is not None:
            current_error = self._white_distance
        elif self._yellow_distance is not None:
            current_error = self._yellow_distance
        self._angular_velo = 0

        p = self.gain * current_error
        d = self.d_gain * (current_error - self.prev_error)/1 # time is 0.1 as refresh at 10 Hz
        self.prev_error = current_error
        self._angular_velo = p + d
        return self._angular_velo
    
    def calculate_pid_control(self, **kwargs):
        # Proportional Integral Derivative
        current_error = 0
        if self.error is not None:
            current_error = self.error
        elif self._white_distance is not None:
            current_error = -self._white_distance
        elif self._yellow_distance is not None:
            current_error = self._yellow_distance
        self._angular_velo = 0

        p = self.gain * current_error

        if(np.abs(self.sum_of_error + current_error*0.1) < 1):
            self.sum_of_error = (self.sum_of_error +  current_error*0.1)
        else:
            self.sum_of_error = np.sign(self.sum_of_error + current_error*0.1)*1
        i = self.i_gain * self.sum_of_error

        d = self.d_gain * (current_error - self.prev_error)/1 # time is 0.1 as refresh at 10 Hz

        self.prev_error = current_error
        self._angular_velo = p + d + i
        return self._angular_velo

    def moveForward(self, desired_rotations):
        rate = rospy.Rate(20)
        self._direction_left = self._direction_right = 1
        self._vel_right = THROTTLE * self._direction_left
        self._vel_left = 0.74 * THROTTLE
        message = WheelsCmdStamped(vel_left=self._vel_left, vel_right=self._vel_right)

        self._init_ticks_left = self._ticks_left
        self._init_ticks_right = self._ticks_right

        while not rospy.is_shutdown():
            if self._ticks_right is not None and self._ticks_left is not None:
                self._publisher.publish(message)
                if np.absolute(self._ticks_left - self._init_ticks_left)/self._resolution_left > desired_rotations:
                    # Stop moving
                    self._vel_left = 0
                    self._vel_right = 0
                    message = WheelsCmdStamped(vel_left=self._vel_left, vel_right=self._vel_right)
                    self._publisher.publish(message)
                    rospy.loginfo_once(f"Finished moving forward")
                    break_time = 1
                    time.sleep(break_time)
                    break
            rate.sleep()

    def sharpTurn(self, dir):
        rate = rospy.Rate(20)
        turn_throttle = 0.8*THROTTLE
        if dir == 1:
            self._direction_left = 1
            self._direction_right = -1
        else:
            self._direction_left = -1
            self._direction_right = 1
        self._vel_left = turn_throttle*self._direction_left
        self._vel_right = turn_throttle*self._direction_right
        message = WheelsCmdStamped(vel_left=self._vel_left, vel_right=self._vel_right)

        self._init_ticks_left = self._ticks_left
        self._init_ticks_right = self._ticks_right

        while not rospy.is_shutdown():
            if self._ticks_right is not None and self._ticks_left is not None:
                self._publisher.publish(message)
                if np.absolute(self._ticks_left - self._init_ticks_left)/self._resolution_left > (1/3 - 0.125):
                    # Stop moving
                    self._vel_left = 0
                    self._vel_right = 0
                    message = WheelsCmdStamped(vel_left=self._vel_left, vel_right=self._vel_right)
                    self._publisher.publish(message)
                    rospy.loginfo_once(f"Finished 90 degree turn")
                    break_time = 1
                    time.sleep(break_time)
                    break
            rate.sleep()

    def on_shutdown(self):
        # Stop the Duckiebot if shutdown
        message = Twist2DStamped(v=0, omega=0)
        self.movement_publisher.publish(message)

    def run(self):
        while not rospy.is_shutdown():
            # if self._white_distance is not None and self._yellow_distance is not None and self.error is not None:
                # print(f"From camera_distortion.py: White distance {self._white_distance}, Yellow distance {self._yellow_distance}, Error: {self.error}")
            # Movement with P/PD/PID Controller
            
            if self.state == "MOVING":
                if self.bot_area < 5000:
                    signal = 0
                    if CONTROLLER == "P":
                        signal = self.calculate_p_control()
                    elif CONTROLLER == "PD":
                        signal = self.calculate_pd_control()
                    else:
                        signal = self.calculate_pid_control()

                    # Publish message to Duckiebot to move with linear velocity self._linear_velo and angular velocity 'signal'
                    message = Twist2DStamped(v=self._linear_velo, omega=signal)
                    print(f"Omega: {signal}, Error = {self.error}")
                    self.movement_publisher.publish(message)
                else: 
                    # Bot area is above threshold; STOP MOVEMENT
                    self.state = "STOPPED"
                    rospy.loginfo("Bot command override enabled.")
                    self.prev_time = time.time()
                    message = Twist2DStamped(v=0, omega=0)
                    self.movement_publisher.publish(message)

            # Stopped long enough; proceed to 'manual controlled' stage        
            elif self.state == "STOPPED" and time.time() - self.prev_time >= 3:
                self.state = "OVERRIDE"
                # Move with pre-programmed movements
                self.sharpTurn(-1)
                self.moveForward(1)
                self.sharpTurn(1)
                self.moveForward(2)
                self.sharpTurn(1)
                self.moveForward(1)
                self.state = "MOVING"
            rospy.Rate(10).sleep()


if __name__ == '__main__':
    # create the node
    node = LaneBehaviourNode(node_name='lane_behaviour_node')
    node.run()
    # keep spinning
    rospy.spin()
