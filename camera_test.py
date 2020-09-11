"""This is a file to expiriment with just the computer vision parts of this class to quickly test and develop new features."""

from all_seeing_drone.drone import SeeingDrone
import logging

if __name__ == "__main__":
    drone = SeeingDrone(r"C:\Users\Mikey\Videos\droneVideos", logging_level=logging.DEBUG)
    # uncomment the below to use drone camera
    src=r'rtsp://192.168.100.1/cam1/mpeg4'
    # uncomment the below to use webcam camera
    # src = 0
    drone.computer_video_check(src=src, find_distance=True, write_camera_focal_debug=True)