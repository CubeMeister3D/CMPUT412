#!/usr/bin/env python3

import os
import rospy
import numpy as np
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image

import cv2
from cv_bridge import CvBridge

class CameraReadWriteNode(DTROS):

    def __init__(self, node_name):
        # initialize the DTROS parent class
        super(CameraReadWriteNode, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)

        # static parameters
        self._vehicle_name = os.environ['VEHICLE_NAME']
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"

        # bridge between OpenCV and ROS
        # self._bridge = CvBridge()

        # create window
        # self._window = "camera-reader"
        # cv2.namedWindow(self._window, cv2.WINDOW_AUTOSIZE)

        # construct subscriber
        self._subcriber = rospy.Subscriber(self._camera_topic, CompressedImage, self.callback)

        self._annotation_topic = "annotated_image"
        self._publisher = rospy.Publisher(self._annotation_topic, Image, queue_size = 10)

    def callback(self, msg):
        # convert JPEG bytes to CV image
        image = self._bridge.compressed_imgmsg_to_cv2(msg)

        # apply greyscale 
        image_new = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        size = str(image_new.shape[1])+"x"+str(image_new.shape[0]) # image.shape[0] - cols, image.shape[1] - rows

        # annotation of image
        to_annotate_top = "Duck "+self._vehicle_name+" says: "
        to_annotate_bottom = "\'Cheese! Capturing "+size+ " Quack-tastic!\'" 
        image_new = cv2.putText(image_new, to_annotate_top, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (1, 1, 1), 2, cv2.LINE_AA)
        image_new = cv2.putText(image_new, to_annotate_bottom, (50, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (1, 1, 1), 2, cv2.LINE_AA)
        
        # encode new image 
        image_new = self._bridge.cv2_to_imgmsg(image_new, encoding="mono8")

        # cv2.imshow(self._window, image)

        # publish annotated image
        # self._publisher.publish(image_new)
        
        # display frame
        cv2.waitKey(1)

if __name__ == '__main__':
    # create the node
    node = CameraReadWriteNode(node_name='camera_read_write_node')
    # keep spinning
    rospy.spin()
