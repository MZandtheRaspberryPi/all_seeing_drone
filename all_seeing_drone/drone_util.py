import cv2
import CoDrone
import time


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



def get_joystick_buttons(sensor_data, joystick_object):
    sensor_data["button_A"] = joystick_object.get_button(0)
    sensor_data["button_X"] = joystick_object.get_button(2)
    sensor_data["button_B"] = joystick_object.get_button(1)
    sensor_data["button_Y"] = joystick_object.get_button(3)
    return sensor_data


def calibrate():
    """A simple script to calibrate the CoDrone. CoDrone needs calibrating when
    you observe it drifing in the air, ie, unable to stay in one place.

    Ensure your bluetooth device is plugged into the usb port so that
    a connection can be made with the drone.

    From here, place your drone in a spot where it can take off and have some
    clearance around it as it may drift during calibration.
    Then, run the script."""
    drone = CoDrone.CoDrone()
    drone.pair()
    time.sleep(1)

    drone.calibrate()

    time.sleep(7)

    drone.disconnect()
