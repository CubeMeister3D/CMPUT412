#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# launch subscriber
rosrun ex2 odometry_p2.py

# wait for app to end
dt-launchfile-join
