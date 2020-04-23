import time
import logging

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
