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
import CoDrone
from CoDrone import Color, Mode
import cv2
import time
import os
import datetime
import logging
import math
import pygame
import argparse
from threading import Thread
from multiprocessing.pool import ThreadPool


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


class FPS:
    """A class to measures frames per second of a given camera
    and video processing pipeline. from:
    https://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/"""
    def __init__(self):
        # store the start time, end time, and total number of frames
        # that were examined between the start and end intervals
        self._start = None
        self._end = None
        self._numFrames = 0

    def start(self):
        # start the timer
        self._start = datetime.datetime.now()
        return self

    def stop(self):
        # stop the timer
        self._end = datetime.datetime.now()

    def update(self):
        # increment the total number of frames examined during the
        # start and end intervals
        self._numFrames += 1

    def elapsed(self):
        # return the total number of seconds between the start and
        # end interval
        return (self._end - self._start).total_seconds()

    def fps(self):
        # compute the (approximate) frames per second
        return self._numFrames / self.elapsed()

class WebcamVideoStream:
    """A class to read from a camera with threading to speed up fps
    https://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/
    """
    def __init__(self, src=r'rtsp://192.168.100.1/cam1/mpeg4'):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False
    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # return the frame most recently read
        return self.frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
        self.stream.release()


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
    drone = CoDrone.CoDrone()
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



def get_joystick_buttons(sensor_data, joystick_object):
    sensor_data["button_A"] = joystick_object.get_button(0)
    sensor_data["button_X"] = joystick_object.get_button(2)
    sensor_data["button_B"] = joystick_object.get_button(1)
    sensor_data["button_Y"] = joystick_object.get_button(3)
    return sensor_data

get_joystick_buttons_logger = logger(get_joystick_buttons)

def recognize_human(frame_to_analyze):
    start_time = time.time()
    full_body = cv2.data.haarcascades+'haarcascade_fullbody.xml'
    frontal_face = cv2.data.haarcascades+'haarcascade_frontalface_default.xml'
    profile_face = cv2.data.haarcascades+'haarcascade_profileface.xml'
    person_cascade = cv2.CascadeClassifier(frontal_face)
    # Downscale to improve frame_to_analyze rate
    # frame_to_analyze = cv2.resize(frame_to_analyze, (640, 360))
    # Haar-cascade classifier needs a grayscale image
    gray_frame_to_analyze = cv2.cvtColor(frame_to_analyze, cv2.COLOR_RGB2GRAY)
    rects = person_cascade.detectMultiScale(gray_frame_to_analyze)
    for (x, y, w, h) in rects:
        cv2.rectangle(frame_to_analyze, (x, y), (x+w, y+h), (0, 255, 0), 2)
    logging.debug("Analyzed frame and overlayed rectangles in {} seconds".format(time.time() - start_time))
    return frame_to_analyze


def process_video(video_directory, written_video_name):
    PROCESSED_VIDEO_NAME = written_video_name + "_processed"
    full_body = cv2.data.haarcascades+'haarcascade_fullbody.xml'
    frontal_face = cv2.data.haarcascades+'haarcascade_frontalface_default.xml'
    profile_face = cv2.data.haarcascades+'haarcascade_profileface.xml'

    person_cascade = cv2.CascadeClassifier(frontal_face)


    cap = cv2.VideoCapture(os.path.join(video_directory, written_video_name) + ".avi")
    # this is stuff for saving video
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(os.path.join(video_directory, PROCESSED_VIDEO_NAME + '.avi'),
                          fourcc, 6, (int(width), int(height)))

    while (cap.isOpened()):
        r, frame = cap.read()
        if r:
            start_time = time.time()
            # Downscale to improve frame rate
            # frame = cv2.resize(frame, (640, 360))
            # Haar-cascade classifier needs a grayscale image
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            rects = person_cascade.detectMultiScale(gray_frame)

            print("FPS: ", 1.0 / (time.time() - start_time))  # FPS = 1 / time to process loop
            end_time = time.time()
            print("Elapsed Time:", end_time - start_time)

            for (x, y, w, h) in rects:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # don't need to see the frame here
            # cv2.imshow("preview", frame)
            out.write(frame)
        if not r:
            break
    print("wrote processed video")
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def command_top_gun(drone_object, joystick_object, data_to_update, exit_flag):
    tolerance = 20
    if not exit_flag:
        # drone_object.set_yaw(int(joystick_object.get_axis(0)*100))
        # drone_object.set_throttle(int(joystick_object.get_axis(1)*100)*(-1))
        # drone_object.set_pitch(int(joystick_object.get_axis(3)*100)*(-1))
        # drone_object.set_roll(int(joystick_object.get_axis(4)*100))
        # drone_object.move()
        yaw = int(joystick_object.get_axis(0) * 100)
        throttle = int(joystick_object.get_axis(1) * 100) * (-1)
        pitch = int(joystick_object.get_axis(3) * 100) * (-1)
        roll = int(joystick_object.get_axis(4) * 100)

        data_to_update["sent_yaw"] = 0 if not abs(yaw) > tolerance else yaw
        data_to_update["sent_throttle"] = 0 if not abs(throttle) > tolerance else throttle
        data_to_update["sent_pitch"] = 0 if not abs(pitch) > tolerance else pitch
        data_to_update["sent_roll"] = 0 if not abs(roll) > tolerance else roll

        # now = time.time()
        # drone_object.set_yaw(data_to_update["sent_yaw"])
        # drone_object.set_throttle(data_to_update["sent_throttle"])
        # drone_object.set_pitch(data_to_update["sent_pitch"])
        # drone_object.set_roll(data_to_update["sent_roll"])
        # logging.debug("sending data to drone took {} seconds".format(time.time() - now))
        # drone_object.set_yaw(0 if not abs(yaw) > tolerance else yaw)
        # drone_object.set_throttle(0 if not abs(throttle) > tolerance else throttle)
        # drone_object.set_pitch(0 if not abs(pitch) > tolerance else pitch)
        # drone_object.set_roll(0 if not abs(roll) > tolerance else roll)
        now = time.time()
        drone_object.move(data_to_update["sent_roll"], data_to_update["sent_pitch"],
                          data_to_update["sent_yaw"], data_to_update["sent_throttle"])
        logging.debug("moving took {} seconds".format(time.time() - now))
        # print("Yaw {} throttle {} pitch {} roll {}".format(
        #       0 if not abs(yaw) > 20 else yaw,
        #       0 if not abs(throttle) > 20 else throttle,
        #       0 if not abs(pitch) > 20 else pitch,
        #       0 if not abs(roll) > 20 else roll)
    return data_to_update

command_top_gun_logger = logger(command_top_gun)

def hud_display(frame, width, height, data_to_update, font, font_scale, font_thickness):
    """function to take a frame and layor on sensor information and return it"""
    half_width = int(width/2)
    half_height = int(height/2)

    # drawing a border around image. Just for looks.
    cv2.rectangle(img=frame, pt1=(int(width/40), int(height/40)),
                  pt2=(int(width*39/40), int(height*39/40)),
                  color=(0, 255, 0), thickness=1)

    # horizon line for tracking roll. Will adjust depending on roll.
    cv2.line(frame, pt1=(int(width*3/10), int(height/2 + data_to_update["sent_roll"])),
             pt2=(int(width*7/10), int(height/2 - data_to_update["sent_roll"])),
             color=(0, 255, 0), thickness=1)

    # version 1 of the arrow for tracking pitch
    # cv2.line(frame, pt1=(int(width/2 - width*1/20), int(height/2 + height*1/20 - data_to_update["sent_pitch"])),
    #          pt2=(int(width/2), int(height/2)),
    #          color=(0, 255, 0), thickness=1)
    # cv2.line(frame, pt1=(int(width/2 + width*1/20), int(height/2 + height*1/20 - data_to_update["sent_pitch"])),
    #          pt2=(int(width/2), int(height/2)),
    #          color=(0, 255, 0), thickness=1)
    # version 2 of the arrow for tracking pitch. Will adjust to show Pitch.
    cv2.line(frame, pt1=(int(width/2 - width*1/20), int(height/2 + height*1/20 - data_to_update["sent_pitch"])),
             pt2=(int(width/2), int(height/2 - data_to_update["sent_pitch"])),
             color=(0, 255, 0), thickness=1)
    cv2.line(frame, pt1=(int(width/2 + width*1/20), int(height/2 + height*1/20 - data_to_update["sent_pitch"])),
             pt2=(int(width/2), int(height/2 - data_to_update["sent_pitch"])),
             color=(0, 255, 0), thickness=1)
    spacing_between_text = 2
    roll_string = "Roll: {}".format(data_to_update["sent_roll"])
    roll_size = cv2.getTextSize(roll_string, font, font_scale, font_thickness)[0][1] + spacing_between_text
    pitch_string = "Pitch: {}".format(data_to_update["sent_pitch"])
    pitch_size = cv2.getTextSize(pitch_string, font, font_scale, font_thickness)[0][1] + spacing_between_text
    yaw_string = "Yaw: {}".format(data_to_update["sent_yaw"])
    yaw_size = cv2.getTextSize(yaw_string, font, font_scale, font_thickness)[0][1] + spacing_between_text
    throttle_string = "Throttle: {}".format(data_to_update["sent_throttle"])
    throttle_size = cv2.getTextSize(throttle_string, font, font_scale, font_thickness)[0][1] + spacing_between_text
    battery_string = "Battery: {}%".format(data_to_update["battery_percentage"])
    battery_size = cv2.getTextSize(battery_string, font, font_scale, font_thickness)[0][1] + spacing_between_text

    x_location = int(width/40 * 2)
    y_location = int(height - (battery_size + throttle_size + \
                 yaw_size + pitch_size + roll_size) - (width/40 * 2))

    y_location += roll_size
    cv2.putText(frame, roll_string, (x_location, y_location), font, font_scale,
                (0, 255, 0), font_thickness, cv2.LINE_AA)
    y_location += pitch_size
    cv2.putText(frame, pitch_string, (x_location, y_location), font, font_scale,
                (0, 255, 0), font_thickness, cv2.LINE_AA)

    y_location += yaw_size
    cv2.putText(frame, yaw_string, (x_location, y_location), font, font_scale,
                (0, 255, 0), font_thickness, cv2.LINE_AA)

    y_location += throttle_size
    cv2.putText(frame, throttle_string, (x_location, y_location), font, font_scale,
                (0, 255, 0), font_thickness, cv2.LINE_AA)

    y_location += battery_size
    cv2.putText(frame, battery_string, (x_location, y_location), font, font_scale,
                (0, 255, 0), font_thickness, cv2.LINE_AA)

    return frame


hud_display_logger = logger(hud_display)


def update_sensor_information(drone_object, data_to_update):
    # from drones eyes
    # throttle is up/down, yaw is rotate left/right. Often on left joystick.
    # pitch is move forward/back, roll is move left/right. often on right joy.
    # data_to_update["roll"] = drone_object.get_roll()
    # data_to_update["throttle"] = drone_object.get_throttle()
    # data_to_update["pitch"] = drone_object.get_pitch()
    # data_to_update["yaw"] = drone_object.get_yaw()
    # gets x and y coordinates from optical sensor
    # doesn't take into account rotation
    # data_to_update["opt_flow_position"] = drone_object.get_opt_flow_position()
    # # print(position.X, position.Y)
    # data_to_update["trim"] = drone_object.get_trim()
    # # print(trim.ROLL, trim.PITCH, trim.YAW, trim.THROTTLE)
    # # from ir sensor, in mm
    # data_to_update["height"] = drone_object.get_height()
    # # gets x, y, z, in m/s^2
    # data_to_update["acceleration"] = drone_object.get_accelerometer()
    # print(acceleration.X, acceleration.Y, acceleration.Z)
    # gets data from gyro sensor to get roll, pitch, yaw, as angles
    data_to_update["gyro_angles"] = drone_object.get_gyro_angles()
    # print(GyroAngles.ROLL, GyroAngles.PITCH, GyroAngles.YAW)
    # gets data from gyro sensor for angular speed, roll, pitch, yaw
    # data_to_update["gyro_speed"] = drone_object.get_angular_speed()
    # print(gyrodata.ROLL, gyrodata.PITCH, gyrodata.YAW)
    return data_to_update

update_sensor_information_logger = logger(update_sensor_information)

def update_non_real_time_info(drone_object, sensor_data):
    # splitting this into another sensor as needs to be run less frequently
    # I have my doubts about how accurate battery % is
    sensor_data["battery_percentage"] = drone_object.get_battery_percentage()
    # sensor_data["battery_voltage"] = drone_object.get_battery_voltage()
    # sensor_data["internal_temperature"] = drone_object.get_drone_temp()
    # sensor_data["state"] = drone_object.get_state()
    # returns data from barometer sensor
    # sensor_data["pressure"] = drone_object.get_pressure()
    return sensor_data

update_non_real_time_info_logger = logger(update_non_real_time_info)

if __name__ == "__main__":
    main()
