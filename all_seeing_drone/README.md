# all_seeing_drone
This is a project to work with the RoboLink CoDrone (https://www.robolink.com/codrone/) to enable the drone to follow someone. It introduces computer vision and movements off of the computer vision to work towards the CoDrone autonomously following someone.

![Demo of Autonomous Drone](drone.gif)  
Bottom left: processed drone video with debug information  
Top left: cell phone camera in corner of room, showing drone flying  
Right: raw drone input video  
Notice it tracking me!

Most of this project needs the CoDrone's bluetooth module with drivers plugged into the computer (to command the drone), as well as the computer to be connected to the CoDrone's wifi (to get the video stream). To use this, setup a virtual python environment using the requirements.txt file, and then use that environment in a jupyter notebook context using Drone Demo.ipynb, or simply run the run_drone.py script in Python.

This is distributed with a GNU General Public License v3.0, permitting almost any use except distributing closed source versions.