"""This is a file to test a main run of the drone."""

from all_seeing_drone.drone import SeeingDrone
import logging

if __name__ == "__main__":
    drone = SeeingDrone("C:\\Users\\Mikey\\Videos\\droneVideos\\", logging_level=logging.INFO)
    drone.activate_drone(find_face=True, launch=True)
