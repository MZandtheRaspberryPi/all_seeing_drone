"""This is a file to test a main run of the drone."""

from all_seeing_drone.drone import SeeingDrone
import logging

if __name__ == "__main__":
    drone = SeeingDrone("C:\\Users\\Mikey\\Videos\\droneVideos\\", logging_level=logging.DEBUG)
    # tracker and keep distance don't play nice together, so watch out and make sure if one is true the other is false
    drone.activate_drone(find_face=True, launch=True, use_tracker=True, keep_distance=True)
