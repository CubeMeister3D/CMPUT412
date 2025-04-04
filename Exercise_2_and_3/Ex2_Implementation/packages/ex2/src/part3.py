#!/usr/bin/env python3

import os
import rospy
import numpy as np
import time
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import WheelEncoderStamped, WheelsCmdStamped, LEDPattern
from std_msgs.msg import ColorRGBA

# throttle and direction for each wheel
THROTTLE = 0.5   
DESIRED_ROTATIONS = 6      # desired distance

class WheelOdometryHandler(DTROS):
    def __init__(self, node_name):
        # initialize the DTROS parent class
        super(WheelOdometryHandler, self).__init__(node_name=node_name, node_type=NodeType.PERCEPTION)
        # static parameters
        self._vehicle_name = os.environ['VEHICLE_NAME']

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
        """
        #color1
        color1 = ColorRGB()
        color1.r = 255.0
        color1.g = 0.0
        color1.b = 0.0
        color1.a = 255.0
        self._color1_msg = LEDPattern(rgb_vals = [color1,color1,color1,color1])
        #color2
        
        color2 = ColorRGB()
        color2.r = 0.0
        color2.g = 0.0
        color2.b = 255.0
        color2.a = 255.0
        self._color2_msg = LEDPattern(rgb_vals = [color2,color2,color2,color2])
        led_topic = f"/{self._vehicle_name}/led_emitter_node/led_pattern"
        self._led_publisher = rospy.Publisher(led_topic, LEDPattern, queue_size = 2)
        """
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

    def sharpTurn(self):
        rate = rospy.Rate(20)
        turn_throttle = 0.8*THROTTLE
        self._direction_left = 1
        self._direction_right = -1
        self._vel_left = turn_throttle*self._direction_left
        self._vel_right = turn_throttle*self._direction_right
        message = WheelsCmdStamped(vel_left=self._vel_left, vel_right=self._vel_right)

        self._init_ticks_left = self._ticks_left
        self._init_ticks_right = self._ticks_right

        while not rospy.is_shutdown():
            if self._ticks_right is not None and self._ticks_left is not None:
                self._publisher.publish(message)
                if np.absolute(self._ticks_left - self._init_ticks_left)/self._resolution_left > (1/3 - 0.2):
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

    def arcTurn(self):
        rate = rospy.Rate(20)
        turn_throttle = 0.8*THROTTLE
        self._direction_left = self._direction_right = 1

        self._vel_left = turn_throttle*self._direction_left
        self._vel_right = 0.6*turn_throttle*self._direction_right

        message = WheelsCmdStamped(vel_left=self._vel_left, vel_right=self._vel_right)

        self._init_ticks_left = self._ticks_left
        self._init_ticks_right = self._ticks_right

        while not rospy.is_shutdown():
            if self._ticks_right is not None and self._ticks_left is not None:
                self._publisher.publish(message)
                if np.absolute(self._ticks_left - self._init_ticks_left)/self._resolution_left > (3-0.5):
                    # Stop moving
                    self._vel_left = 0
                    self._vel_right = 0
                    message = WheelsCmdStamped(vel_left=self._vel_left, vel_right=self._vel_right)
                    self._publisher.publish(message)
                    rospy.loginfo_once(f"Finished arc turn")
                    break_time = 1
                    time.sleep(break_time)
                    break
            rate.sleep()

    def run(self):
        # Publish commands to wheels 20x/sec based on recorded status
        # self._led_publisher.publish(self._color1_msg)
        time.sleep(5)
        # self._led_publisher.publish(self._color2_msg)
        self.moveForward(6) # 1.25m / 
        self.sharpTurn()
        self.moveForward(4) # 36 in
        self.arcTurn()
        self.moveForward(3) # 24 in (short)
        self.arcTurn()
        self.moveForward(4) # 36 in
        self.sharpTurn()
        time.sleep(5)
        # self._led_publisher.publish(self._color1_msg)
        rospy.loginfo_once("Finished path. Shutting down.")
        rospy.signal_shutdown("Finished job.")

    
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
