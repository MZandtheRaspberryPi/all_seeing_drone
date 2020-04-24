import cv2
import datetime
from threading import Thread
from multiprocessing.pool import ThreadPool
import logging
import time


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
        (self._grabbed, self._frame) = self.stream.read()

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False
    def start(self):
        # start the thread to read frames from the video stream
        self.fps_cam = FPS().start()
        self.fps_act = FPS().start()
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            # otherwise, read the next frame from the stream
            (self._grabbed, self._frame) = self.stream.read()
            self.fps_cam.update()

    def read(self):
        # return the frame most recently read
        return self._frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
        self.fps_cam.stop()
        self.fps_act.stop()
        elapsed_cam = self.fps_cam.elapsed()
        fps_cam = self.fps_cam.fps()
        elapsed_act = self.fps_act.elapsed()
        fps_act = self.fps_act.fps()
        time.sleep(.5)
        self.stream.release()
        return elapsed_cam, fps_cam, elapsed_act, fps_act

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
