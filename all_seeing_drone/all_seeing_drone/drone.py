description = """
drone_recognize_human.py is a script that allows control of a drone and
records the drone video overlaying sensor information, and finally processes
the video and attempts to recognize humans. The end goal is to be able to have
it fly autonomously and notify the user if anyone is home.

Still some ways to go on this :)

Make sure to follow these steps.

1.Connect FPV module to CoDrone
2.Conect BLE module to computer make sure to reset it (RED LED)
3.Connect XBOX 360 Controller to your computer via USB
4.Connect Wifi from Computer to FPV (ex: PETRONE FPV 1f57 ,PASSWORD 12345678)
5.Run this program from the command line

Can be run normally with the below command that will recognize humans post-landing
 python drone_recognize_human.py -post_process

Or can be run to recognize humans in real time with the below:
 python drone_recognize_human.py -recognize_human

Or can be run without recognizing humans (faster run time/highest FPS):
 python drone_recognize_human.py

Can be profiled (for speed testing) by running with the below, or enabling
debug logging as there is useful logging built in:
 python -m cProfile -s cumtime drone_recognize_human.py

"""

# TODO: figure out drone drift
# TODO: figure out speed issues with frames including downsampling
from CoDrone import CoDrone
import cv2
import time
import os
import datetime
import logging
import pygame
import argparse


from all_seeing_drone.drone_camera import FPS, WebcamVideoStream, recognize_human, process_video
from all_seeing_drone.drone_util import get_joystick_buttons, update_non_real_time_info, update_sensor_information, hud_display
from all_seeing_drone.drone_movement import command_top_gun
from all_seeing_drone.drone_util import calibrate

def logger(func):
    def log_func(*args):
        # logging.debug(
        #     'Running "{}" with arguments {}'.format(func.__name__, args))
        start_time = time.time()
        return_values = func(*args)
        duration = round(time.time() - start_time, 6)
        # logging.debug("Ran {} with arguments {} in {} sec".format(func.__name__,
        #                                                          args,
        #                                                          duration))
        logging.debug("Ran {} in {} sec".format(func.__name__, duration))
        return return_values
    # Necessary for closure to work (returning WITHOUT parenthesis)
    return log_func

# video starts and ends with a button
# takeoff and land with a button
# control with joystick in mid air
# get FPV and overlay info like battery, signal (if possible), angular direction
# x,y, ect
# button to change leds/light show
# threading to handle video processing in parrallel?
# flags like is flying and such to control when to quit

class SeeingDrone(CoDrone):
    """A class to enable computer vision for the robolink CoDrone"""
    def __init__(self, video_output_dir=None, post_process=False, recognize_human=False):
        """
        video_output_dir: str the directory on your computer to output videos to
        post_process: bool whether to analyze the video after drone flight is done
        recognize_human: bool whether to look for faces in the video
        """
        super().__init__()

        self.video_output_dir = video_output_dir
        self.time_now = datetime.datetime.now()
        self.video_name = self.time_now.strftime('drone_%Y%m%d_%H%M')
        self.video_full_path = os.path.join(self.video_output_dir, self.video_name) + ".avi"

        # setting up logging
        self.log_path = self.video_output_dir
        self.file_name = self.video_name + '.txt'
        self.full_path = os.path.join(self.log_path, self.file_name)
        # creating log file if it doesn't yet exist
        if not os.path.exists(self.full_path):
            logFile = open(self.full_path, 'w')
            logFile.write("")
            logFile.close()
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
            handlers=[
                logging.FileHandler(self.full_path),
                logging.StreamHandler()
            ])

        self.fps_actual = 29.0

        # setting up gamepad
        try:
            pygame.init()
            pygame.joystick.init()
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            self.axes = self.joystick.get_numaxes()
        except:
            logging.info("No joysticks found.")

        # https://forum.robolink.com/topic/148/how-to-control-my-drone-with-opencv
        # https://drive.google.com/drive/folders/1mpqUnG6tBHZDdtv_FG5Z0npH_5DAT785



        # this is stuff for saving video, used to get  width and height from stream
        # but now could make it class variables if wanted as using threaded class
        # width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        # height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.width = 480
        self.height = 320
        logging.debug("Height: {} Width: {}".format(self.height, self.width))
        #create the font
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = .3
        self.font_thickness = int(1)

    def connect(self):
        logging.info("Pairing")
        self.pair()
        logging.info("Paired")
        time.sleep(1)
        # show that got connection
        # gamepad.set_vibration(1, 1, 1000)
        i = 0
        while not self.is_ready_to_fly():
            logging.info("drone not ready to fly")
            logging.info("sleeping")
            i += 1
            time.sleep(1)
            if i == 10:
                self.shutdown()
                raise Exception("Drone slept for 10 seconds, killing script")

        self.exit_flag = False
        self.take_off = False
        self.sensor_data = {}
        self.sensor_data_counter = 0
        self.exit_duration = 0

    @staticmethod
    def calibrate_drone():
        """A simple script to calibrate the CoDrone. CoDrone needs calibrating when
        you observe it drifing in the air, ie, unable to stay in one place.

        Ensure your bluetooth device is plugged into the usb port so that
        a connection can be made with the drone.

        From here, place your drone in a spot where it can take off and have some
        clearance around it as it may drift during calibration.
        Then, run the script."""
        calibrate()

    def _setup_camera(self):
        # Capture video from the Wifi Connection to FPV module
        # RTSP =(Real Time Streaming Protocol)
        # TODO: eventually make the connection switch automatic
        self.vs = WebcamVideoStream().start()

    def show_camera(self):
        self._setup_camera()
        print("Staring Camera, press q to quit with camera window in focus. "
              "If it doesn't close, check caps lock and num lock.", flush=True)
        self.frame_list = []
        while True:
            self.frame_list.append(self.vs.read())
            # displaying the frame and writing it
            cv2.imshow("Drone Camera", self.frame_list[-1])
            # updating fps counter
            self.vs.fps_act.update()
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
        self._shutdown_camera()
        self.shutdown()

    def _shutdown_camera(self):
        self.elapsed_cam_time, self.cam_fps, self.elapsed_act_time, self.act_fps = self.vs.stop()
        logging.info("elasped cam time: {:.2f}".format(self.elapsed_cam_time))
        logging.info("approx. cam FPS: {:.2f}".format(self.cam_fps))
        logging.info("elasped act time: {:.2f}".format(self.elapsed_act_time))
        logging.info("approx. act FPS: {:.2f}".format(self.act_fps))
        cv2.destroyAllWindows()
        # https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html
        # fourcc format and extension (.avi) is particular. expirement
        # to see what works on your computer
        print("Writing out video to {}".format(self.video_full_path))
        self.fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.out = cv2.VideoWriter(self.video_full_path, self.fourcc,
                                   self.act_fps, (int(self.width),
                                   int(self.height)))
        for frame in self.frame_list:
            self.out.write(frame)
        self.out.release()

    def shutdown(self):
        logging.info("shutting down")
        self.disconnect()







def main():
    # setting up logging
    log_path = r'.'
    file_name = 'droneLog.txt'
    full_path = os.path.join(log_path, file_name)
    # creating log file if it doesn't yet exist
    if not os.path.exists(full_path):
        logFile = open(full_path, 'w')
        logFile.write("")
        logFile.close()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler(full_path),
            logging.StreamHandler()
        ])
    # logging.disable(logging.DEBUG)
    # setting up command line arguments
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-post_process", action="store_true",
                        help="""switch to determine whether to run open cv2's
                        haar cascade methodology to recognize a face after
                        saving the original video. If included, will do.""")
    parser.add_argument("-recognize_human", action="store_true",
                        help="""switch to determine whether to run open cv2's
                        haar cascade methodology to recognize a face in
                        real time as flying. If included, will do.""")
    args = parser.parse_args()

    TIME_NOW = datetime.datetime.now()
    # set your output directory to save videos in the below:
    VIDEO_DIRECTORY = "non_existant_directory"
    VIDEO_NAME = TIME_NOW.strftime('drone_capture_%Y_%m_%d_%H-%M')

    # setting up gamepad
    pygame.init()
    pygame.joystick.init()
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    axes = joystick.get_numaxes()

    # creating drone object and pairing
    # ensure wifi is connected to drone for camera
    drone = CoDrone()
    logging.info("Pairing")
    drone.pair()
    logging.info("Paired")

    # show that got connection
    # gamepad.set_vibration(1, 1, 1000)
    # giving connection a moment to resolve
    time.sleep(1)

    while not drone.is_ready_to_fly():
        logging.info("drone not ready to fly")
        logging.info("sleeping")
        time.sleep(1)

    # https://forum.robolink.com/topic/148/how-to-control-my-drone-with-opencv
    # https://drive.google.com/drive/folders/1mpqUnG6tBHZDdtv_FG5Z0npH_5DAT785

    # Capture video from the Wifi Connection to FPV module
    # RTSP =(Real Time Streaming Protocol)
    # TODO: eventually make the connection switch automaticS
    vs = WebcamVideoStream(src=r'rtsp://192.168.100.1/cam1/mpeg4').start()
    fps = FPS().start()

    # this is stuff for saving video, used to get  width and height from stream
    # but now could make it class variables if wanted as using threaded class
    # width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    # height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = 480
    height = 320
    logging.debug("Height: {} Width: {}".format(height, width))


    #create the font
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = .3
    font_thickness = int(1)
    # https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html
    # fourcc format and extension (.avi) is particular. expirement
    # to see what works on your computer
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(os.path.join(VIDEO_DIRECTORY, VIDEO_NAME) + ".avi",
                          fourcc, 6, (int(width), int(height)))

    exit_flag = False
    take_off = False
    sensor_data = {}
    sensor_data_counter = 0
    exit_duration = 0
    while True:
        print("Press button A to take off; Button B to land; Button X to kill camera and post-process; and Button Y to emergency stop", flush=True)
        loop_start_time = time.time()
        now = time.time()
        # r, frame = cap.read()
        frame = vs.read()
        # updating fps counter
        fps.update()
        logging.debug("Reading frame took {} seconds".format(time.time() - now))

        # incrementing sensor_data, only read non-real time
        # sensors every 20th frame. I defined which to not check each loop
        # based on what I wanted real time updates on
        # sensor_data = update_sensor_information_logger(drone, sensor_data)
        if sensor_data_counter % 20 == 0:
            logging.debug("updating non real time info")
            sensor_data = update_non_real_time_info_logger(drone, sensor_data)
            logging.debug("Battery: {}".format(sensor_data["battery_percentage"]))
            sensor_data_counter = 0

        sensor_data_counter += 1

        # refreshing data from joystick
        now = time.time()
        pygame.event.get()
        logging.debug("Getting events from pygame took {} seconds".format(time.time() - now))

        # sending joystick commands, only if drone is in flight
        sensor_data = command_top_gun_logger(drone, joystick, sensor_data, exit_flag)

        # overlaying HUD on the frame
        frame = hud_display_logger(frame, width, height, sensor_data, font, font_scale, font_thickness)

        # recognizing people
        if args.recognize_human:
            frame = recognize_human(frame)

        # reading the buttons
        sensor_data = get_joystick_buttons_logger(sensor_data, joystick)

        now = time.time()

        # button A is take off
        if sensor_data["button_A"] and not take_off:
            logging.info("taking off")
            Thread(target=drone.takeoff, args=()).start()
            take_off = True
        # button B is land
        elif sensor_data["button_B"] and take_off:
            logging.info("landing")
            Thread(target=drone.land, args=()).start()
            take_off = False
        # button X is kill camera, start post-processing
        elif sensor_data["button_X"]:
            logging.info("killing camera")
            exit_flag = True
        # button Y is emergency stop
        elif sensor_data["button_Y"]:
            logging.info("emergency stopping")
            drone.emergency_stop()
            take_off = False
            break

        logging.debug("Evaluating buttons/doing commands if buttons took {} seconds".format(time.time() - now))

        # displaying the frame and writing it
        now = time.time()
        cv2.imshow("frame", frame)
        out.write(frame)
        logging.debug("Writing/showing frame took {} seconds".format(time.time() - now))
        logging.debug("full loop took {} seconds".format(time.time() - loop_start_time))

        if exit_flag:
            # stop the timer and display FPS information
            fps.stop()
            logging.info("elasped time: {:.2f}".format(fps.elapsed()))
            logging.info("approx. FPS: {:.2f}".format(fps.fps()))
            break


    drone.disconnect()
    vs.stop()
    out.release()
    cv2.destroyAllWindows()

    if args.post_process:
           process_video(VIDEO_DIRECTORY, VIDEO_NAME)




get_joystick_buttons_logger = logger(get_joystick_buttons)








command_top_gun_logger = logger(command_top_gun)

hud_display_logger = logger(hud_display)


update_sensor_information_logger = logger(update_sensor_information)


update_non_real_time_info_logger = logger(update_non_real_time_info)

if __name__ == "__main__":
    main()
