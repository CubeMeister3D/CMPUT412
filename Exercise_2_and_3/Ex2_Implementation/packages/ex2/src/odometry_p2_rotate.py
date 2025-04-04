#!/usr/bin/env python3

import os
import rospy
import numpy as np
import time
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import WheelEncoderStamped
from duckietown_msgs.msg import WheelsCmdStamped

# throttle and direction for each wheel
THROTTLE = 0.35
DESIRED_ROTATIONS = 1.1/3

class WheelOdometryHandler(DTROS):
    def __init__(self, node_name):
            # initialize the DTROS parent class
            super(WheelOdometryHandler, self).__init__(node_name=node_name, node_type=NodeType.PERCEPTION)
            # static parameters
            self._vehicle_name = os.environ['VEHICLE_NAME']
            self._left_encoder_topic = f"/{self._vehicle_name}/left_wheel_encoder_node/tick"
            self._right_encoder_topic = f"/{self._vehicle_name}/right_wheel_encoder_node/tick"

            self._wheels_topic = f"/{self._vehicle_name}/wheels_driver_node/wheels_cmd"
            # temporary data storage
            self._ticks_left = None
            self._ticks_right = None

            self._init_ticks_left = None
            self._init_ticks_right = None

            self._resolution_left = None
            self._resolution_right = None

            # form the message for wheel publisher
            self._direction_left = 1
            self._direction_right = -1
            self._vel_left = THROTTLE * self._direction_left 
            self._vel_right = THROTTLE * self._direction_right

            # construct publisher
            self._publisher = rospy.Publisher(self._wheels_topic, WheelsCmdStamped, queue_size=1)

            # construct subscriber
            self.sub_left = rospy.Subscriber(self._left_encoder_topic, WheelEncoderStamped, self.callback_left)
            self.sub_right = rospy.Subscriber(self._right_encoder_topic, WheelEncoderStamped, self.callback_right)

    def callback_left(self, data):
        # log general information once at the beginning
        rospy.loginfo_once(f"Left encoder resolution: {data.resolution}")
        rospy.loginfo_once(f"Left encoder type: {data.type}")
        # store data value
        self._ticks_left = data.data

        if not self._init_ticks_left:
            self._init_ticks_left = data.data
            rospy.loginfo_once(f"Left Init: {self._init_ticks_left}")
            self._resolution_left = data.resolution

    def callback_right(self, data):
        # log general information once at the beginning
        rospy.loginfo_once(f"Right encoder resolution: {data.resolution}")
        rospy.loginfo_once(f"Right encoder type: {data.type}")
        # store data value
        self._ticks_right = data.data

        if not self._init_ticks_right:
            self._init_ticks_right = data.data
            rospy.loginfo_once(f"Right Init: {self._init_ticks_right}")
            self._resolution_right = data.resolution

    def run(self):
        # publish received tick messages every 0.05 second (20 Hz)
        rate = rospy.Rate(20)
        message = WheelsCmdStamped(vel_left=self._vel_left, vel_right=self._vel_right)
        went_fwd = False

        while not rospy.is_shutdown():
            if self._ticks_right is not None and self._ticks_left is not None and self._init_ticks_left is not None:
                self._publisher.publish(message)
                # Publish moving 90 deg
                if np.absolute(self._ticks_left - self._init_ticks_left)/self._resolution_left > DESIRED_ROTATIONS:
                    # Stop moving
                    self._vel_left = 0
                    self._vel_right = 0
                    message = WheelsCmdStamped(vel_left=self._vel_left, vel_right=self._vel_right)
                    self._publisher.publish(message)

                    rospy.loginfo_once(f"Finished turning")
                    if went_fwd:
                        # Stopping for 2nd time when program is finished
                        rospy.signal_shutdown("Finished job.")
                    elif not went_fwd:
                        # Prevent from throttling an infinite amount of times
                        self._direction_left = -self._direction_left
                        self._direction_right = -self._direction_right
                        self._vel_left = self._direction_left * THROTTLE 
                        self._vel_right = self._direction_right * THROTTLE

                        self._init_ticks_left = self._ticks_left
                        self._init_ticks_right = self._ticks_right
                        message = WheelsCmdStamped(vel_left=self._vel_left, vel_right=self._vel_right)
                        went_fwd = True
                        rospy.loginfo_once("Turning back to original pose")
                        
                        break_time = 1
                        time.sleep(break_time)
                
            rate.sleep()
    
    def on_shutdown(self):
        stop = WheelsCmdStamped(vel_left=0, vel_right=0)
        self._publisher.publish(stop)


if __name__ == '__main__':
    # create the node
    node = WheelOdometryHandler(node_name='wheel_control_node')
    # run node
    node.run()
    # keep the process from terminating
    rospy.spin()