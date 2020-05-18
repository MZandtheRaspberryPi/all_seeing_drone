import time
import logging
import cv2
from simple_pid import PID

class DroneController():
    """A class to implement a controller, given a frame and a rectangular bounding box for an object in the frame.
    Uses a PID controller."""
    def __init__(self, setpoint_throttle=-60.0, setpoint_yaw=0.0):
        # throttle setpoint negative so that head will be in top of frame, as bboxy - frame - will be negative
        # pid for x-axis, or yaw adjustments
        # Kp, Ki, Kd are gain constants on the principal, integral, and derivative terms
        # setpoint is the pixel x distance from center of frame to center of object
        # output limits are the limits to CoDrone's yaw
        # sample_time is minimum time that this will update. here, it'll be bounded naturally by the FPS
        # of the computer vision system, but putting in a limit in case other folks run on upgraded hardware where that
        # isn't a bottleneck
        # TODO: can these handle negative errors and negative outputs?
        self.setpoint_throttle = setpoint_throttle
        self.setpoint_yaw = setpoint_yaw
        self.yaw_pid = PID(Kp=.5, Ki=0.0, Kd=0.0, setpoint=self.setpoint_throttle, sample_time=round(1/14, 2), output_limits=(-100, 100))
        self.throttle_pid = PID(Kp=.1, Ki=0.0, Kd=0.0, setpoint=self.setpoint_yaw, sample_time=round(1/14, 2), output_limits=(-100, 100))
        self.throttle_errors = []
        self.throttle_outputs = []
        self.yaw_errors = []
        self.yaw_outputs = []

    @staticmethod
    def calc_x_y_error(frame, bounding_box):
        """This takes a frame object and a tuple of a bounding box where the entries are coordinates of
         top left and bottom right corners, ie, (x1, y1, x2, y2) and outputs the x distance from object center to
         frame center and y distance from object center to frame center"""
        # calculate center of image
        width, height = frame.shape[:2]
        frame_center_coords = (width/2, height/2)
        bb_x_center = (bounding_box[2] - bounding_box[0])/2 + bounding_box[0]
        bb_y_center = (bounding_box[3] - bounding_box[1])/2 + bounding_box[1]
        bb_center_coords = (bb_x_center, bb_y_center)
        x_distance_from_center = frame_center_coords[0] - bb_center_coords[0]
        # making it so if frame center is below object center, result is negative so that pid tries to increase throttle
        y_distance_from_center = bb_center_coords[1] - frame_center_coords[1]
        return x_distance_from_center, y_distance_from_center

    def get_throttle_and_yaw(self, frame, bounding_box, write_frame_debug_info=False):
        x_error, y_error = DroneController.calc_x_y_error(frame, bounding_box)
        self.throttle_errors.append(y_error)
        self.yaw_errors.append(x_error)
        throttle_output = self.throttle_pid(y_error)
        yaw_output = self.yaw_pid(x_error)
        self.throttle_outputs.append(throttle_output)
        self.yaw_outputs.append(yaw_output)
        if write_frame_debug_info:
            self.write_debug(frame, x_error, y_error, throttle_output, yaw_output)
        return int(throttle_output), int(yaw_output)

    def write_debug(self, frame, x_error, y_error, throttle_output, yaw_output, font=cv2.FONT_HERSHEY_SIMPLEX,
                    color=(0, 255, 0), font_scale=.3, font_thickness=1):
        components_throttle = self.throttle_pid.components
        components_yaw = self.yaw_pid.components
        throttle_text = "Throttle: {} (y_error: {})".format(throttle_output, y_error)
        throttle_component_text = "Throttle components: {}".format(components_throttle)
        yaw_text = "Yaw: {} (x_error: {})".format(yaw_output, x_error)
        yaw_component_text = "Yaw components: {}".format(components_yaw)
        cv2.putText(frame, throttle_text, (0, 20),
                    font, font_scale, color, font_thickness)
        cv2.putText(frame, throttle_component_text, (0, 30),
                    font, font_scale, color, font_thickness)
        cv2.putText(frame, yaw_text, (0, 40),
                    font, font_scale, color, font_thickness)
        cv2.putText(frame, yaw_component_text, (0, 50),
                    font, font_scale, color, font_thickness)

    def reset(self):
        self.throttle_pid.reset()
        self.yaw_pid.reset()


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
