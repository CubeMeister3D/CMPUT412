#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# launch subscriber
rosrun ex3 camera_distortion.py

# wait for app to end
dt-launchfile-join
