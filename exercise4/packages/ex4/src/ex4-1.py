#!/usr/bin/env python3

# import required libraries
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
from dt_apriltags import Detector

class AprilNode(DTROS):
    def __init__(self, node_name):
        super(AprilNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        # add your code here

        self._vehicle_name = os.environ['VEHICLE_NAME']

        #get the topics for the camera
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        self._camera_info_topic = f"/{self._vehicle_name}/camera_node/camera_info"
        self._grey_image = "grey_image"
        self._contour_image = "contour_image"
        self._bridge = CvBridge()

        #matrix for the undistortion
        self._camera_mtx = None
        self._proj_mtx = None
        self._d_params = None
        # From Duckiebot Portal Homography Matrix
        self.H = np.array([[-7.386314935629145e-05, 0.0008023555341693569, 0.37599498192426706],
                          [-0.002291756610528657, 2.68205480472991e-05, 0.7395497492359417],
                          [-0.00024216376950167147, 0.016003927930200265, -2.832489619617473]])
        self.H_inv = np.linalg.inv(self.H)

        #camera subscriber and publishers
        self._image_info_subcriber = rospy.Subscriber(self._camera_info_topic, CameraInfo, self.info_callback)
        self._image_subcriber = rospy.Subscriber(self._camera_topic, CompressedImage, self.saveImage)
        self._grey_publisher = rospy.Publisher(self._grey_image, Image, queue_size = 10)      
        self._contour_publisher = rospy.Publisher(self._contour_image, Image, queue_size = 10)

        #AprilTag definitions
        self.detector = Detector(families="tag36h11")
        #Get the LED topic
        vehicle_name = os.environ['VEHICLE_NAME']
        led_topic = f"/{vehicle_name}/led_emitter_node/led_pattern"
        self._led_publisher = rospy.Publisher(led_topic, LEDPattern, queue_size = 3)
        #define four colours
        self._white = ColorRGBA(255.0,255.0,255.0,255.0)
        self._blue = ColorRGBA(0.0,0.0,255.0,255.0)
        self._green = ColorRGBA(0.0,255.0,0.0,255.0)
        self._red = ColorRGBA(255.0,0.0,0.0,255.0)

        #get the topics for the wheels
        self._wheels_topic = f"/{self._vehicle_name}/wheels_driver_node/wheels_cmd" # Wheel command topic for publisher
        self._wheel_publisher = rospy.Publisher(self._wheels_topic, WheelsCmdStamped, queue_size=1)

        self._last_image = None
        self._average_line_distance = 0
        self._find_line = True
        self._stop_time = 0.5

    def use_leds(self, tagID=None):
        color = self._white
        if tagID != None:
            if tagID == "21" or tagID == "22":
                self._stop_time = 3
                color = self._red
            elif tagID == "93" or tagID == "94":
                self._stop_time = 1
                color = self._green
            elif tagID == "133" or "15":
                self._stop_time = 2
                color = self._blue
        self._led_publisher.publish(LEDPattern(rgb_vals = [color,color,color,color,color]))

    def find_april_tag(self, image):
        #convert jpeg to cv2 image
        height, width, _ = image.shape
        image_new = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        results = self.detector.detect(image_new)
        #convert image back to RGB
        rgb_new = cv2.cvtColor(image_new, cv2.COLOR_GRAY2RGB)
        #find create boxes around the tag
        for r in results:
            tagID = str(r.tag_id)
            self.use_leds(tagID)
            (ptA, ptB, ptC, ptD) = r.corners
            ptA = (int(ptA[0]),int(ptA[1]))
            ptB = (int(ptB[0]),int(ptB[1]))
            ptC = (int(ptC[0]),int(ptC[1]))
            ptD = (int(ptD[0]),int(ptD[1]))

            cv2.line(rgb_new, ptA, ptB, (0,255,0),2)
            cv2.line(rgb_new, ptB, ptC, (0,255,0),2)
            cv2.line(rgb_new, ptC, ptD, (0,255,0),2)
            cv2.line(rgb_new, ptD, ptA, (0,255,0),2)

            #create the text for tag number
            font = cv2.FONT_HERSHEY_SIMPLEX
            textSize = cv2.getTextSize(tagID,font,0.7,2)[0]
            origin = (int(r.center[0]-textSize[0]/2), int(r.center[1]+textSize[1]/2))
            rgb_new = cv2.putText(rgb_new, tagID, origin, font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        image_new = self._bridge.cv2_to_imgmsg(rgb_new, encoding="rgb8")
        self._grey_publisher.publish(image_new)

    def pixel_to_world(self, pixel_x, pixel_y):
        pixel = np.array([pixel_x, pixel_y, 1])  # Homogeneous coordinates
        ground = np.dot(self.H, pixel)
        ground /= ground[2]
        return ground

    def lane_dimension(self, contour):
        x, y, w, h = cv2.boundingRect(contour)  # (x, y) = top-left corner, (w, h) = width & height
        distance = y
        return distance

    def detectRedLine(self, image):
        #crop the image
        height, width, _ = image.shape
        ratio = 0.45
        height_clip = int(height * (0.5))
        width_clip = int(((width - width*0.70)/2))
        # Crop and blur
        cropped = image[height_clip:height, width_clip:width - width_clip]
        blurred = cv2.blur(cropped, (3,3))
        # Set range for red color and 
        # define mask 
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        red_lower = np.array([172, 125, 50], np.uint8) 
        red_upper = np.array([179, 255, 255], np.uint8) 
        red_lower2 = np.array([0, 125, 50], np.uint8)
        red_upper2 = np.array([5, 255, 255], np.uint8)
        red_mask = cv2.inRange(hsv_image, red_lower, red_upper) 
        red_mask2 = cv2.inRange(hsv_image, red_lower2, red_upper2)
        red_mask = cv2.bitwise_or(red_mask, red_mask2) # to capture both higher and lower HSV values on the OpenCV reds
        kernel = np.ones((5, 5), "uint8") 
        red_mask = cv2.dilate(red_mask, kernel) 
        # Creating contour to track red color 
        contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 

        max_contour = None
        max_area = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_contour = contour
        distance = self.lane_dimension(max_contour)
        line_distance = distance
        self._average_line_distance = line_distance
        #cv2.drawContours(blurred, max_contour, -1, (0, 0, 255), 2)
        print(f"Color RED lane distance: {distance}, average: {self._average_line_distance}")
        
        #image_new = self._bridge.cv2_to_imgmsg(blurred, encoding="rgb8")
        #self._contour_publisher.publish(image_new)

    def moveStraight(self):
        pass

    def imageCallBack(self, image):
        image = self._bridge.compressed_imgmsg_to_cv2(image)
        # Undistort image
        if self._camera_mtx is not None and self._proj_mtx is not None and self._d_params is not None:
            image = cv2.undistort(image, self._camera_mtx, self._d_params)
        if self._stop_time == 0.5:
            self.find_april_tag(image)
        if (self._find_line):
            self.detectRedLine(image)


    # define other functions if needed
    def run(self):
        rospy.sleep(2)  # wait for the node to initialize
        rate = rospy.Rate(50)
        self._led_publisher.publish(LEDPattern(rgb_vals = [self._white,self._white,self._white,self._white,self._white]))
        while not rospy.is_shutdown():
            if self._last_image is not None:
                self.imageCallBack(self._last_image)
            if self._average_line_distance > 400:
                self._average_line_distance = 0
                self._find_line = False
                stop = WheelsCmdStamped(vel_left=0, vel_right=0)
                self._wheel_publisher.publish(stop)
                time.sleep(self._stop_time)
                self._wheel_publisher.publish(WheelsCmdStamped(vel_left=0.4,vel_right=0.5))
                self.use_leds()
            else:
                self._wheel_publisher.publish(WheelsCmdStamped(vel_left=0.4,vel_right=0.5))
            rate.sleep()
        # add your code here
        # call the functions you have defined above for executing the movements

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
    
    def saveImage(self, image):
        self._last_image = image

if __name__ == '__main__':
    # initiate the AprilTag node
    aprilNode = AprilNode(node_name = "april_tag_node")
    aprilNode.run()
    # call the function run of class MoveNode
    rospy.spin()
