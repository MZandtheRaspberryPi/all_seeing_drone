from all_seeing_drone.drone_camera import DroneVision

import time
import logging
import cv2
from simple_pid import PID

class DroneController():
    """A class to implement a controller, given a frame and a rectangular bounding box for an object in the frame.
    Uses a PID controller."""
    def __init__(self, frame, keep_distance=False, setpoint_throttle=-60.0, setpoint_yaw=0.0, meter_distance=2.0, ctr_rect_proportions=(3/16, 3/16)):
        """rect_proportions is a tuple with the width of the X part of rectangle as first, and width of y as second"""
        # flag to know if we're controlling distance too
        self.keep_distance = keep_distance
        # throttle setpoint negative so that head will be in top of frame, as bboxy - frame - will be negative
        # pid for x-axis, or yaw adjustments
        # Kp, Ki, Kd are gain constants on the principal, integral, and derivative terms
        # setpoint is the pixel x distance from center of frame to center of object
        # output limits are the limits to CoDrone's yaw
        # sample_time is minimum time that this will update. here, it'll be bounded naturally by the FPS
        # of the computer vision system, but putting in a limit in case other folks run on upgraded hardware where that
        # isn't a bottleneck
        self.setpoint_throttle = setpoint_throttle
        self.setpoint_yaw = setpoint_yaw
        if keep_distance:
            self.meter_distance = meter_distance
            self.pitch_pid = PID(Kp=5.0, Ki=0.5, Kd=1, setpoint=0.0, sample_time=round(1 / 14, 2),
                                 output_limits=(-10, 10))
        self.yaw_pid = PID(Kp=.1, Ki=0.1, Kd=0.2, setpoint=self.setpoint_throttle, sample_time=round(1/14, 2), output_limits=(-15, 15))
        # because the drone is heavy with camera going down is much easier than up
        # reflecting this in output limits in the throttle pid
        self.throttle_pid = PID(Kp=.5, Ki=0.3, Kd=0.4, setpoint=self.setpoint_yaw, sample_time=round(1/14, 2), output_limits=(-20, 80))
        # flags to determine when to reset the pid if nescessary
        # for instance, when our error is within tolerance for throttle, we'll reset the throttle PID
        # when we're back out of tolerance we'll use it again
        self.throttle_pid_on = True
        self.yaw_pid_on = True
        if keep_distance:
            self.pitch_pid_on = True

        self.throttle_errors = []
        self.throttle_outputs = []
        self.yaw_errors = []
        self.yaw_outputs = []
        self.pitch_errors = []
        self.pitch_outputs = []

        # setting a center rectangle where if a face is in that area, we won't be out of tolerance
        self.height, self.width = frame.shape[:2]
        x1 = int((self.width/2) - (self.width * ctr_rect_proportions[0]))
        x2 = int((self.width/2) + (self.width * ctr_rect_proportions[0]))
        y1 = int((self.height/2) - (self.height * ctr_rect_proportions[1]))
        y2 = int((self.height/2) + (self.height * ctr_rect_proportions[1]))
        self.center_rectangle_coords = (x1, y1, x2, y2)
        self.min_abs_x_error = (self.center_rectangle_coords[2] - self.center_rectangle_coords[0]) / 2
        self.min_abs_y_error = (self.center_rectangle_coords[3] - self.center_rectangle_coords[1]) / 2

        # setting minimum distance error in meters. The current approach for estimating distance is rather inaccurate.
        # so, setting this high.
        if keep_distance:
            self.min_distance_error = .5

    @staticmethod
    def calc_x_y_error(frame, bounding_box):
        """This takes a frame object and a tuple of a bounding box where the entries are coordinates of
         top left and bottom right corners, ie, (x1, y1, x2, y2) and outputs the x distance from object center to
         frame center and y distance from object center to frame center
         """
        # calculate center of image
        height, width = frame.shape[:2]
        frame_center_coords = (width/2, height/2)
        bb_x_center = (bounding_box[2] - bounding_box[0])/2 + bounding_box[0]
        bb_y_center = (bounding_box[3] - bounding_box[1])/2 + bounding_box[1]
        bb_center_coords = (bb_x_center, bb_y_center)
        x_distance_from_center = frame_center_coords[0] - bb_center_coords[0]
        # making it so if frame center is below object center, result is negative so that pid tries to increase throttle
        y_distance_from_center = bb_center_coords[1] - frame_center_coords[1]
        return x_distance_from_center, y_distance_from_center

    def _get_yaw_from_x_error(self, x_error):
        if abs(x_error) >= self.min_abs_x_error:
            yaw_output = self.yaw_pid(x_error)
            components_yaw = self.yaw_pid.components
            if not self.yaw_pid_on:
                self.yaw_pid_on = True
        else:
            yaw_output = 0
            components_yaw = ("no_error", "no_error", "no_error")
            if self.yaw_pid_on:
                self.yaw_pid_on = False
                self.yaw_pid.reset()
        return yaw_output, components_yaw

    def _get_throttle_from_y_error(self, y_error):
        if abs(y_error) > self.min_abs_y_error:
            throttle_output = self.throttle_pid(y_error)
            components_throttle = self.throttle_pid.components
            if not self.throttle_pid_on:
                self.yaw_pid_on = True
        else:
            throttle_output = 0
            components_throttle = ("no_error", "no_error", "no_error")
            if self.throttle_pid_on:
                self.throttle_pid_on = False
                self.throttle_pid.reset()
        return throttle_output, components_throttle

    def _get_pitch_from_distance(self, distance_error):
        if abs(distance_error) > self.min_distance_error:
            pitch_output = self.pitch_pid(distance_error)
            components_pitch = self.pitch_pid.components
            if not self.pitch_pid_on:
                self.pitch_pid_on = True
        else:
            pitch_output = 0.0
            components_pitch = ("no_error", "no_error", "no_error")
            if self.pitch_pid_on:
                self.pitch_pid_on = False
                self.pitch_pid.reset()
        return pitch_output, components_pitch

    def get_drone_movements(self, frame, bounding_box, estimate_distance=False, write_frame_debug_info=False):
        x_error, y_error = DroneController.calc_x_y_error(frame, bounding_box)
        self.throttle_errors.append(y_error)
        self.yaw_errors.append(x_error)

        yaw_output, components_yaw = self._get_yaw_from_x_error(x_error)
        throttle_output, components_throttle = self._get_throttle_from_y_error(y_error)

        if estimate_distance:
            frame, distance = DroneVision.calculate_distance(frame, bounding_box)
            distance_error = self.meter_distance - distance
            self.pitch_errors.append(distance_error)
            pitch_output, components_pitch = self._get_pitch_from_distance(distance_error)
        else:
            pitch_output = 0.0
            components_pitch = ("dist. False", "dist. False", "dist. False")
            distance_error = 0.0

        self.throttle_outputs.append(throttle_output)
        self.yaw_outputs.append(yaw_output)
        self.pitch_outputs.append(pitch_output)
        if write_frame_debug_info:
            frame = self.write_debug(frame, x_error, y_error, distance_error, throttle_output,
                                     yaw_output, pitch_output, components_throttle, components_yaw, components_pitch)
        return int(throttle_output), int(yaw_output), int(pitch_output), frame

    def write_debug(self, frame, x_error, y_error, distance_error, throttle_output, yaw_output, pitch_output, components_throttle,
                    components_yaw, components_pitch, font=cv2.FONT_HERSHEY_SIMPLEX,
                    color=(0, 255, 0), font_scale=.3, font_thickness=1):
        throttle_text = "Throttle: {} (y_error: {})".format(round(throttle_output, 2), round(y_error, 2))
        throttle_component_text = "Throttle components: {}".format([round(component, 2) if not isinstance(component, str) else component for component in components_throttle])
        yaw_text = "Yaw: {} (x_error: {})".format(round(yaw_output, 2), round(x_error, 2))
        yaw_component_text = "Yaw components: {}".format([round(component, 2) if not isinstance(component, str) else component for component in components_yaw])
        pitch_text = "Pitch: {} (distance_error: {})".format(round(pitch_output, 2), round(distance_error, 2))
        pitch_components = "Pitch components: {}".format([round(component, 2) if not isinstance(component, str) else component for component in components_pitch])

        cv2.putText(frame, throttle_text, (0, 20),
                    font, font_scale, color, font_thickness)
        cv2.putText(frame, throttle_component_text, (0, 30),
                    font, font_scale, color, font_thickness)
        cv2.putText(frame, yaw_text, (0, 40),
                    font, font_scale, color, font_thickness)
        cv2.putText(frame, yaw_component_text, (0, 50),
                    font, font_scale, color, font_thickness)
        cv2.putText(frame, pitch_text, (0, 60),
                    font, font_scale, color, font_thickness)
        cv2.putText(frame, pitch_components, (0, 70),
                    font, font_scale, color, font_thickness)

        # writing center rectangle
        cv2.rectangle(frame, (self.center_rectangle_coords[0], self.center_rectangle_coords[1]),
                      (self.center_rectangle_coords[2], self.center_rectangle_coords[3]),
                      color=color)
        return frame

    def reset(self):
        self.throttle_pid.reset()
        self.yaw_pid.reset()
        if self.keep_distance:
            self.pitch_pid.reset()


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
