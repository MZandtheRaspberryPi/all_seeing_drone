import CoDrone
import time

drone = CoDrone.CoDrone()
drone.pair()
time.sleep(1)

drone.calibrate()

time.sleep(7)

drone.disconnect()
