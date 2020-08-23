"""
drone.py is a library that allows a CoDrone to be autonomous and follow a face.

Make sure to follow these steps.

1.Connect FPV module to CoDrone
2.Conect BLE module to computer make sure to reset it (RED LED)
4.Connect Wifi from Computer to FPV (ex: PETRONE FPV 1f57 ,PASSWORD 12345678)
5.Use the class as you want, or use the script run_drone.py up a level in the repo to get an idea for how to use the class.

Some useful links:
# https://forum.robolink.com/topic/148/how-to-control-my-drone-with-opencv
# https://drive.google.com/drive/folders/1mpqUnG6tBHZDdtv_FG5Z0npH_5DAT785
"""

from all_seeing_drone.drone_camera import WebcamVideoStream, DroneVision
from all_seeing_drone.drone_movement import DroneController
from all_seeing_drone.drone_util import calibrate

from CoDrone import CoDrone
from CoDrone.receiver import Header, DataType
from CoDrone.protocol import Control
import cv2
import numpy as np
import os
import datetime
import logging
import time
from threading import Thread
import pygame
import imutils
import subprocess


class SeeingDrone(CoDrone):
    """A class to enable computer vision for the robolink CoDrone"""
    def __init__(self, video_output_dir: str = None, logging_level: int = logging.INFO):
        """
        :param video_output_dir: str the directory on your computer to output videos to, and the log files
        :param logging_level: how detailed should the logs be.
        """
        super().__init__()

        self.video_output_dir = video_output_dir
        self.time_now = datetime.datetime.now()
        self.video_name = self.time_now.strftime('drone_%Y%m%d_%H%M')
        self.video_full_path = os.path.join(self.video_output_dir, self.video_name) + ".avi"
        self.raw_video_name = self.time_now.strftime('drone_%Y%m%d_%H%M_raw')
        self.full_path_raw = os.path.join(self.video_output_dir, self.raw_video_name) + '.avi'

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
            level=logging_level,
            format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
            handlers=[
                logging.FileHandler(self.full_path)
                # logging.StreamHandler()
            ])

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

        #create the font
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = .3
        self.font_thickness = int(1)

        self.rectangle_thickness = 2

        # this can be used to speed up the deep neural network recognition, by reducing it further if needed
        # making images smaller means less pixels and less operations to perform
        self.resize_image_width = 300

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

    def _setup_camera(self, src=r'rtsp://192.168.100.1/cam1/mpeg4', setup_face_finder=False, setup_tracker=False,
                      min_confidence=.8, tracker_model="kcf"):
        """A function to start the video stream from the drone in a separate thread. The seperate thread will
        continuously grab and store frames so that the main thread can grab frames whenever its free and not wait."""
        # if want to connect to drone, ensure connected to drone's wifi
        if src == r'rtsp://192.168.100.1/cam1/mpeg4':
            wifi_name = subprocess.check_output("netsh wlan show interfaces")
            wifi_name_lower = wifi_name.lower()
            if 'petrone' not in wifi_name_lower.decode():
                self.disconnect()
                raise Exception("to setup drone camera, must be connected to drone wifi. You're connected to:\n\n {}"
                                "\n\nand we expected a name that when made lowercase has 'petrone' in it".format(wifi_name))
        # Capture video from the Wifi Connection to FPV module
        # RTSP =(Real Time Streaming Protocol)
        # TODO: eventually make the connection switch automatic
        self.vs = WebcamVideoStream(src=src).start()
        self.exit_flag = False
        if setup_face_finder:
            self.drone_detector = DroneVision(min_confidence, setup_tracker=setup_tracker, tracker_model=tracker_model)


    def computer_video_check(self, use_tracker=True, src=0, find_distance=False, write_camera_focal_debug=False):
        """ This is a function to test the drone's computer vision system (or a webcam on the computer) without actually
        launching the drone. This is convenient because drone can crash into things, so its nice to isolate.

        :param use_tracker: this is the flag on whether to use a seperate tracker algorithm when
            the CNN has found a face but then loses it. It can help a drone follow you, but the bounding box from the
            tracker argument can be weird shapes and throw off the drone estimating how far it is from a face
        :param src: this is the webcam to connect to. an int like 0 will connect to your computers webcam, whereas a string like
            r'rtsp://192.168.100.1/cam1/mpeg4' will connect to the drone's camera.
        :param find_distance: whether to estimate distance in the frame. Note this function was calibrated with the CoDrone's
            Camera and so wouldn't be reliable with the computer webcam.
        :param write_camera_focal_debug: whether to write debug info to the frame that will help in calibrating the distance
            estimation function.
        :return:
        """

        self._setup_camera(setup_face_finder=True, src=src, setup_tracker=use_tracker)
        # self.frame_list = []
        self.processed_frame_list = [self.vs.frame_list[-1]]
        print("entering loop, press q to quit. If this doesn't work make sure video frame is in focus,"
              " and caps lock and num lock are off.")
        while True:
            start_time = time.time()
            frame = self.vs.read()
            # self.frame_list.append(frame)
            # resize the frame to speed up calculations
            frame = imutils.resize(frame, width=self.resize_image_width)
            frame, bbox_list = self.drone_detector.detect_and_track(frame, use_tracker=use_tracker, font=self.font, color=(0, 0, 255),
                                                                     rect_thickness=self.rectangle_thickness,
                                                                     font_scale=self.font_scale, font_thickness=self.font_thickness)
            if len(bbox_list) > 0 and find_distance:
                frame, distance = DroneVision.calculate_distance(frame, bbox_list[0])

            if len(bbox_list) > 0 and write_camera_focal_debug:
                (x1, y1, x2, y2) = bbox_list[0]
                height, width = frame.shape[:2]

                x_bbox_len = x2 - x1
                y_bbox_len = y2 - y1

                cv2.putText(frame, "x_bbox_len is {} pixels".format(x_bbox_len), (0, height - 40),
                            self.font, self.font_scale, (0, 0, 255), self.font_thickness)
                cv2.putText(frame, "y_bbox_len is {} pixels".format(y_bbox_len), (0, height - 30),
                            self.font, self.font_scale, (0, 0, 255), self.font_thickness)


            # showing FPS on Frame
            seconds = time.time() - start_time
            frame = self.vs.show_fps(frame, seconds, (0, 10), self.font, self.font_scale,
                                     (0, 255, 0), self.font_thickness)
            self.processed_frame_list.append(frame)
            # displaying the frame and writing it
            cv2.imshow("Webcam Camera", self.processed_frame_list[-1])
            # updating fps counter
            self.vs.fps_act.update()
            if cv2.waitKey(1) & 0xFF == ord('q') or self.exit_flag:
                break
        self._shutdown_camera(self.processed_frame_list)


    def _find_face(self, frame, use_tracker=False):
        """A function to use open cv's deep neural network capability and a pre-trained model to detect
         faces in a frame. It uses a Single Shot Detector (SSD) with a Res Net base network and a KCF Tracker.

         The logic here is really, what to do when you find a face, versus when you don't find a face.

         If the CNN found a face in the previous frame, but didn't in this frame, maybe you want to use a tracker.

         Conversely, if the CNN didn't find a face and the tracker lost it, reinitialize the tracker and such."""
        start_time = time.time()
        # if a bounding box exists use the tracker method as its faster than the dnn method (but less accurate probably)
        if self.tracker_bb is not None and use_tracker:
            logging.debug("using tracker")
            ok, self.tracker_bb = self.tracker.update(frame)
            logging.debug("got result from tracker in {} seconds".format(time.time() - start_time))
            if ok:
                # Tracking success
                p1 = (int(self.tracker_bb[0]), int(self.tracker_bb[1]))
                p2 = (int(self.tracker_bb[0] + self.tracker_bb[2]), int(self.tracker_bb[1] + self.tracker_bb[3]))
                cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
                self.tracker_count += 1
                if self.tracker_count > 180:
                    self.tracker_count = 0
                    self.tracker_bb = None
                return frame
            else:
                self.tracker_bb = None
        logging.debug("using dnn")
        # grab the frame dimensions and convert it to a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0,
                                     (w, h), (104.0, 177.0, 123.0))
        logging.debug("made blob in {} seconds".format(time.time() - start_time))
        # pass the blob through the network and obtain the detections and
        # predictions
        start_time = time.time()
        self.net.setInput(blob)
        detections = self.net.forward()
        logging.debug("got result from dnn in {} seconds".format(time.time() - start_time))
        if detections.shape[2] == 0:
            self.tracker_bb = None
            return frame
        # loop over the detections
        loop_start_time = time.time()
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
            if use_tracker:
                start_time = time.time()
                self.tracker.init(frame, self.tracker_bb)
                logging.debug("initialized tracker in {} seconds".format(time.time() - start_time))

            # draw the bounding box of the face along with the associated
            # probability
            text = "{:.2f}%".format(confidence * 100)
            y = self.tracker_bb[1] - 10 if self.tracker_bb[1] - 10 > 10 else self.tracker_bb[1] + 10
            cv2.rectangle(frame, (self.tracker_bb[0], self.tracker_bb[1]), (self.tracker_bb[2], self.tracker_bb[3]),
                          (0, 0, 255), 2)
            cv2.putText(frame, text, (self.tracker_bb[0], y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        logging.debug("finished loop in {} seconds".format(time.time() - loop_start_time))

        return frame

    def autonomous_move(self):
        """overridding codrone's move command to be instantaneous like the arduino version and not infinite"""
        self.send_control(*self._control.getAll())

    def send_control(self, roll, pitch, yaw, throttle):
        """This function sends control request. Using a seperate version than CoDrone classes as this finishes faster
        and doesn't delay the main loop. Its more like the arduino version in that regard that its instantaneous.

        Args:
            roll: the power of the roll, which is an int from -100 to 100
            pitch: the power of the pitch, which is an int from -100 to 100
            yaw: the power of the yaw, which is an int from -100 to 100
            throttle: the power of the throttle, which is an int from -100 to 100

        Returns: True if responds well, false otherwise.
        """
        header = Header()

        header.dataType = DataType.Control
        header.length = Control.getSize()

        control = Control()
        control.setAll(roll, pitch, yaw, throttle)

        receiving_flag = self._storageCount.d[DataType.Attitude]

        self._transfer(header, control)
        time.sleep(0.02)
        if self._storageCount.d[DataType.Attitude] == receiving_flag:
            self._print_error(">> Failed to send control.")

        return self._storageCount.d[DataType.Attitude] == receiving_flag

    def _move_to_center_person(self, throttle, yaw, pitch):
        """A movement command that won't delay the main loop, and will finish quickly. This is like some of the
        arduino movement functions that are instantaneous, as I had some trouble doing that in the python functions."""
        self.set_throttle(throttle)
        self.set_yaw(yaw)
        self.set_pitch(pitch)
        self.autonomous_move()
        self.set_throttle(0)
        self.set_yaw(0)
        self.set_pitch(0)


    def launch(self, delay=0.0, height=None):
        """A function to launch the drone. Called via a seperate thread from activate_drone often.
        Delay is an argument to give time to start recording and showing camera"""
        self.launching = True
        time.sleep(delay)
        if not self.isConnected():
            print("connecting...")
            self.connect()
        print("taking off")
        logging.info("taking off")
        self.takeoff()
        if height is not None:
            # only effective between 20 and 1500 millimeters
            self.go_to_height(height)
        else:
            # if the camera is attached, the drone is heavier and 100% of throttle isn't so severe
            self.move(3.0, 0, 0, 0, 80)
        self.in_flight = True
        print("done taking off")
        logging.info("done taking off")
        self.launching = False

    def launch_and_show_camera(self, find_face=False):
        self.show_camera(find_face=find_face, launch=True)
        self.shutdown()

    def setup_drone_controller(self, frame, keep_distance=False):
        self.drone_controller = DroneController(frame, keep_distance=keep_distance)
        self.pid_started = False

    def activate_drone(self, find_face: bool = False, min_confidence: float = .85, launch: bool = False,
                       use_tracker: bool = True, follow_face: bool = True, keep_distance: bool = False):
        """
        :param find_face: this determines whether to try to recognize faces
        :param min_confidence: this is the minimum confidence for the CNN to find a face in an image. 85 + usually works well.
        :param launch: this parameter is the one that controls whether or not the drone actually launches
        :param use_tracker: this is the flag on whether to use a seperate tracker algorithm when
            the CNN has found a face but then loses it. It can help a drone follow you, but the bounding box from the
            tracker argument can be weird shapes and throw off the drone estimating how far it is from a face
        :param follow_face: this determines whether or not the drone should move autonomously to follow the faces found
        :param keep_distance: this determines whether the drone should try to estimate distance and keep that at a steady amount.
        :return:
        """
        self.connect()
        self._setup_camera(setup_face_finder=find_face, setup_tracker=use_tracker,
                           min_confidence=min_confidence, tracker_model="kcf")
        self.processed_frame_list = []
        frame = self.vs.read()
        frame = imutils.resize(frame, width=self.resize_image_width)
        if follow_face:
            self.setup_drone_controller(frame, keep_distance=keep_distance)
        if launch:
            Thread(target=self.launch, args=[2.0]).start()
        print("Starting Camera, press q to quit and land drone with camera window in focus. "
              "If it doesn't close, check caps lock and num lock. \n"
              "Press H to hover to continue flight.", flush=True)

        while True:
            start_time = time.time()
            frame = self.vs.read()
            if find_face:
                # resize the frame to speed up calculations
                frame = imutils.resize(frame, width=self.resize_image_width)
                frame, bbox_list = self.drone_detector.detect_and_track(frame, use_tracker=use_tracker, font=self.font, color=(0, 0, 255),
                                                             rect_thickness=self.rectangle_thickness,
                                                             font_scale=self.font_scale, font_thickness=self.font_thickness)
            if follow_face and len(bbox_list) == 1 and not self.exit_flag and not self.launching:
                logging.debug("found face, bbox list is: {}".format(bbox_list))
                throttle, yaw, pitch, frame = self.drone_controller.get_drone_movements(frame, bbox_list[0],
                                                                                        estimate_distance=keep_distance,
                                                                                        write_frame_debug_info=True)
                logging.debug("throttle is {} yaw is {} pitch is {}".format(throttle, yaw, pitch))
                self.pid_started = True
                # overidden CoDrone move command to be instantaneous
                self._move_to_center_person(throttle, yaw, pitch)
                logging.debug("moved")
            elif len(bbox_list) != 1 and not self.exit_flag and not self.launching:
                self._move_to_center_person(0, 0, 0)
            if self.pid_started and not len(bbox_list) == 1:
                self.pid_started = False
                self.drone_controller.reset()

            # showing FPS on Frame
            seconds = time.time() - start_time
            logging.debug("seconds to process frame is {}".format(seconds))
            frame = self.vs.show_fps(frame, seconds, (0, 10), self.font, self.font_scale,
                                     (0, 255, 0), self.font_thickness)
            self.processed_frame_list.append(frame)
            # displaying the frame and writing it
            cv2.imshow("Drone Camera", self.processed_frame_list[-1])
            # updating fps counter
            self.vs.fps_act.update()
            if cv2.waitKey(1) & 0xFF == ord('q') or self.exit_flag:
                if not self.exit_flag:
                    print("landing")
                    logging.info("landing")
                    Thread(target=self.land, args=[]).start()
                    self.exit_flag = True
                    land_time = time.time()
                if self.exit_flag:
                    if time.time() - land_time > 3:
                        logging.debug("exiting loop")
                        break
            if cv2.waitKey(1) & 0xFF == ord('h'):
                print("hovering")
                self.hover()

        self._shutdown_camera(self.processed_frame_list)
        self.shutdown()


    def _shutdown_camera(self, frame_list):
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
        height, width = frame_list[-1].shape[:2]
        self.out = cv2.VideoWriter(self.video_full_path, self.fourcc,
                                   self.act_fps, (int(width),
                                   int(height)))
        for frame in frame_list:
            self.out.write(frame)
        self.out.release()

        # writing raw video
        print("Writing out raw video to {}".format(self.full_path_raw))
        self.fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        height, width = self.vs.frame_list[-1].shape[:2]
        self.out = cv2.VideoWriter(self.full_path_raw, self.fourcc,
                                   self.cam_fps, (int(width),
                                   int(height)))
        for frame in self.vs.frame_list:
            self.out.write(frame)
        self.out.release()

    def shutdown(self):
        logging.info("shutting down")
        self.disconnect()
