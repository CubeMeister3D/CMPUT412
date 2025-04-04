#!/usr/bin/env python3

# import required libraries
import os
import rospy
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import LEDPattern
from std_msgs.msg import ColorRGBA

class LEDNode(DTROS):
    def __init__(self, node_name):
        super(LEDNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        # add your code here
        vehicle_name = os.environ['VEHICLE_NAME']
        led_topic = f"/{vehicle_name}/led_emitter_node/led_pattern"
        self.color = ColorRGBA(255.0,255.0,255.0,255.0)

        # define other variables
        # initialize ros bag file
        self._publisher = rospy.Publisher(led_topic, LEDPattern, queue_size = 3)
        self._publisher.publish(LEDPattern(rgb_vals = [self.color,self.color,self.color,self.color,self.color]))
        pass
    
    #This function is will set all of the LED's to one solid color
    #color is the ColorRGBA value that the LED's will be set to
    def oneColour(color):
        self.color = color
        self._publisher(message = LEDPattern(rgb_vals = [self.color, self.color, self.color, self.color, self.color]))
    
    #This function will turn the front and back LED's on the left side on with a certain color at a certain frequency
    #signal_color is ColorRGBA
    #speed float32
    def leftSignal(self, signal_color):
        rospy.sleep(2)  # wait for the node to initialize
        self._publisher.publish(LEDPattern(rgb_vals = [signal_color,self.color,self.color,self.color,signal_color]))

    def rightSignal(self, signal_color):
        rospy.sleep(2)  # wait for the node to initialize
        self._publisher.publish(LEDPattern(rgb_vals = [self.color,signal_color,signal_color,signal_color,self.color]))


    def run(self):
        rospy.sleep(2)  # wait for the node to initialize
        flag = 0
        rate = rospy.Rate(1)
        #get the colours set up
        color1 = ColorRGBA(r = 255.0,g=0.0,b=0.0,a=255.0)
        color = ColorRGBA()
        color.r = 0.0
        color.g = 0.0
        color.b = 255.0
        color.a = 255.0
        self._publisher.publish(message)
        rate.sleep()

    def on_shutdown(self):
        pass

if __name__ == '__main__':
    # define class LEDNode
    # call the function run of LEDNode
    led_node = LEDNode(node_name = 'led_control_node')
    led_node.leftSignal(ColorRGBA(r=255.0,g=0.0,b=0.0,a=255.0))
    rospy.sleep(4)
    led_node.rightSignal(ColorRGBA(r=255.0,a=255.0))


    rospy.spin()
