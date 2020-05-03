"""This is a file to expiriment with just the computer vision parts of this class to quickly test and develop new features."""

from all_seeing_drone.drone import SeeingDrone
import logging

if __name__ == "__main__":
    drone = SeeingDrone(r"C:\Users\Mikey\Videos\droneVideos", logging_level=logging.DEBUG)
    drone.computer_video_check()