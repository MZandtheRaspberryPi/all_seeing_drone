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

# TODO: figure out drone drift with controller
# TODO: make flight more flexible rather than just hover
from CoDrone import CoDrone
import cv2
import time
import numpy as np
import os
import datetime
import logging
import time
from threading import Thread
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

        self.connect()

    def connect(self):
        """A function to use CoDrone libraries pair funnction to connect to the nearest drone"""
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
        self.in_flight = False
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

    def _setup_camera(self, setup_face_finder=False, min_confidence=.75):
        """A function to start the video stream from the drone in a separate thread. The seperate thread will
        continuously grab and store frames so that the main thread can grab frames whenever its free and not wait."""
        # Capture video from the Wifi Connection to FPV module
        # RTSP =(Real Time Streaming Protocol)
        # TODO: eventually make the connection switch automatic
        self.vs = WebcamVideoStream().start()
        self.exit_flag = False
        if setup_face_finder:
            print("loading dnn model and weights from disk")
            self.model_path = os.path.join(os.path.dirname(__file__), "opencv_models", "res10_300x300_ssd_iter_140000_fp16.caffemodel")
            self.weights_path = os.path.join(os.path.dirname(__file__), "opencv_models", "deploy.prototxt.txt")
            self.net = cv2.dnn.readNetFromCaffe(self.weights_path, self.model_path)
            self.min_confidence = min_confidence

            print("setting up KCF Tracker")
            # setting up tracker we'll use to track face when dnn can't recognize it
            self.tracker = cv2.TrackerKCF_create()
            self.tracker_bb = None


    def _find_face(self, frame):
        """A function to use open cv's deep neural network capability and a pre-trained model to detect
         faces in a frame. It uses a Single Shot Detector (SSD) with a Res Net base network or a KCF Tracker"""
        # if a bounding box exists use the tracker method as its faster than the dnn method (but less accurate probably)
        if self.tracker_bb is not None:
            ok, self.tracker_bb = self.tracker.update(frame)
            if ok:
                # Tracking success
                p1 = (int(self.tracker_bb[0]), int(self.tracker_bb[1]))
                p2 = (int(self.tracker_bb[0] + self.tracker_bb[2]), int(self.tracker_bb[1] + self.tracker_bb[3]))
                cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
                return frame
            else:
                self.tracker_bb = None
        # grab the frame dimensions and convert it to a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        # pass the blob through the network and obtain the detections and
        # predictions
        self.net.setInput(blob)
        detections = self.net.forward()
        if detections.shape[2] == 0:
            self.tracker_bb = None
            return frame
        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]
            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence < self.min_confidence:
                continue
            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            self.tracker_bb = tuple(box.astype("int"))

            # draw the bounding box of the face along with the associated
            # probability
            text = "{:.2f}%".format(confidence * 100)
            y = self.tracker_bb[1] - 10 if self.tracker_bb[1] - 10 > 10 else self.tracker_bb[1] + 10
            cv2.rectangle(frame, (self.tracker_bb[0], self.tracker_bb[1]), (self.tracker_bb[2], self.tracker_bb[3]),
                          (0, 0, 255), 2)
            cv2.putText(frame, text, (self.tracker_bb[0], y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            # Initialize tracker with first frame and bounding box
            self.tracker.init(frame, self.tracker_bb)
        return frame


    def _move_to_center_person(self):
        # this could return 0 distance if no-one is found
        x_distance_from_center, y_distance_from_center = self._calculate_movements()
        if x_distance_from_center < 5:
            yaw = 0
        if y_distance_from_center < 5:
            throttle = 0
        if x_distance_from_center < 5 and y_distance_from_center < 5:
            return
        self.set_throttle(5)
        self.set_yaw(5)
        print("moving")
        self.move()


    def _calculate_ditsance_from_center(self):
        if self.tracker_bb is None:
            return 0, 0
        # calculate center of image
        frame_x_center, frame_y_center = self.width/2, self.height/2
        bb_x_center = (self.tracker_bb[2] - self.tracker_bb[0])/2 + self.tracker_bb[0]
        bb_y_center = (self.tracker_bb[3] - self.tracker_bb[1])/2 + self.tracker_bb[1]
        x_distance_from_center = frame_x_center - bb_x_center
        y_distnace_from_center = frame_y_center - bb_y_center
        return x_distance_from_center, y_distnace_from_center

    def launch(self, delay=0.0, height=None):
        """Delay to give time to start recording and showing camera"""
        time.sleep(delay)
        if not self.isConnected():
            print("connecting...")
            self.connect()
        print("taking off")
        self.takeoff()
        if height is not None:
            self.go_to_height(height)
        self.in_flight = True

    def launch_and_show_camera(self, find_face=False):
        self.show_camera(find_face=find_face, launch=True)
        self.shutdown()

    def show_camera(self, find_face=False, min_confidence=.6, launch=False):
        self._setup_camera(find_face, min_confidence)
        if launch:
            Thread(target=self.launch, args=[2.0, 1000]).start()
        print("Starting Camera, press q to quit with camera window in focus. "
              "If it doesn't close, check caps lock and num lock.", flush=True)
        self.frame_list = []
        while True:
            start_time = time.time()
            frame = self.vs.read()
            if find_face:
                frame = self._find_face(frame)
            self.frame_list.append(frame)
            # showing FPS on Frame
            seconds = time.time() - start_time
            cv2.putText(self.frame_list[-1], "FPS: {}".format(round(1.0/(seconds if seconds != 0.0 else 1/30))), (0, 10),
                        self.font, self.font_scale, (0, 255, 0), self.font_thickness)
            # displaying the frame and writing it
            cv2.imshow("Drone Camera", self.frame_list[-1])
            # updating fps counter
            self.vs.fps_act.update()
            if cv2.waitKey(30) & 0xFF == ord('q') or self.exit_flag:
                if self.in_flight and not self.exit_flag:
                    print("landing")
                    Thread(target=self.land, args=[]).start()
                    self.exit_flag = True
                    land_time = time.time()
                if self.exit_flag and self.is_flying() == False:
                    if time.time() - land_time > 5:
                        break
            if cv2.waitKey(30) & 0xFF == ord('h'):
                print("hovering")
                self.hover()
        self._shutdown_camera()



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
