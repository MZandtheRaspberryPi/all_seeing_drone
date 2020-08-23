"""This is a file to test a main run of the drone."""

from all_seeing_drone.drone import SeeingDrone
import logging

if __name__ == "__main__":
    drone = SeeingDrone("C:\\Users\\Mikey\\Videos\\droneVideos\\", logging_level=logging.DEBUG)
    # tracker and keep distance don't play super nice together
    # this is because tracker can sometimes distort the bounding box
    # so watch out and make sure if one is true the other is false, or use with caution together
    drone.activate_drone(find_face=True, launch=True, use_tracker=True, keep_distance=True)
