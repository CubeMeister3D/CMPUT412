#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# launch subscriber
rosrun ex2 part3.py

# wait for app to end
dt-launchfile-join
