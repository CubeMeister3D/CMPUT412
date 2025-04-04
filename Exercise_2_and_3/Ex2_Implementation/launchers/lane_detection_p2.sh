#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# launch subscriber
rosrun ex3 lane_detection_p2.py

# wait for app to end
dt-launchfile-join
