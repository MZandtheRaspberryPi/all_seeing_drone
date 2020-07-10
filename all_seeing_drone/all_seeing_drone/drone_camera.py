import cv2
import datetime
from threading import Thread
import logging
import os
import time
import numpy as np


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
    def __init__(self, src=r'rtsp://192.168.100.1/cam1/mpeg4', camera_fps=30):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)
        time.sleep(2)
        (self._grabbed, self._frame) = self.stream.read()
        self.height, self.width = self._frame.shape[0:2]
        logging.debug("Pre-resizing Height: {} Pre-resizing Width: {}".format(self.height, self.width))

        self.camera_fps = camera_fps

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

        # camera read thread will stop if it exceeds this time in seconds
        self.max_run_time = 300

        # will record all frames here
        self.frame_list = []
    def start(self):
        # start the thread to read frames from the video stream
        self.fps_cam = FPS().start()
        self.fps_act = FPS().start()

        self.thread_list = []

        self.thread_list.append(Thread(target=self.update, args=()))
        self.thread_list[0].start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped or a time limit is reached
        thrad_start_time = time.time()
        while True:
            # if the thread indicator variable is set, or the video is running more than a set time, stop the thread
            if self.stopped or (time.time() - thrad_start_time) > self.max_run_time:
                return

            # otherwise, read the next frame from the stream
            (self._grabbed, self._frame) = self.stream.read()
            self.frame_list.append(self._frame)
            self.fps_cam.update()

    def read(self):
        # return the frame most recently read
        return self._frame

    def show_fps(self, frame, elapsed_time, position, font, font_scale, color, font_thickness):
        # if seconds is less than 1/camera fps than we're looping over frames faster than the camera can update
        # so, set fps to 30
        max_fps = 1/self.camera_fps
        if elapsed_time < max_fps:
            fps = round(max_fps, 2)
        else:
            fps = round(1/elapsed_time, 2)
        cv2.putText(frame, "FPS: {}".format(fps), position, font, font_scale, color, font_thickness)
        return frame


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

class DroneTracker():
    """A class to wrap some functionality around open cv's object tracker functionality for the CoDrone"""
    def __init__(self, tracker_model):
        print("setting up {} Tracker".format(tracker_model))
        self.tracker_model = tracker_model
        # https://www.pyimagesearch.com/2018/07/30/opencv-object-tracking/
        # recommends CSRT for slower fps/higher accuracy
        # KCF for higher fps, slightly lower accuracy
        # MOSSE for speed
        # initialize a dictionary that maps strings to their corresponding
        # OpenCV object tracker implementations
        self.OPENCV_OBJECT_TRACKERS = {
            "csrt": cv2.TrackerCSRT_create,
            "kcf": cv2.TrackerKCF_create,
            "boosting": cv2.TrackerBoosting_create,
            "mil": cv2.TrackerMIL_create,
            "tld": cv2.TrackerTLD_create,
            "medianflow": cv2.TrackerMedianFlow_create,
            "mosse": cv2.TrackerMOSSE_create
        }
        assert self.tracker_model in self.OPENCV_OBJECT_TRACKERS.keys(), "tracker_model input must be in {}".format(OPENCV_OBJECT_TRACKERS.keys())
        self.multi_tracker = cv2.MultiTracker()

    def initialize_tracker(self, frame, bounding_box_list):
        success_list = []
        for bbox in bounding_box_list:
            success = self.multi_tracker.add(self.OPENCV_OBJECT_TRACKERS[self.tracker_model](), frame, bbox)
            success_list.append(success)
        return success_list

    def track(self, frame, color=(255, 0, 0), rect_thickness=2, circle_radius=4, circle_thickness=-1):
        # grab the new bounding box coordinates of the object
        (success, boxes) = self.multi_tracker.update(frame)
        # check to see if the tracking was a success
        if success:
            for i, box in enumerate(boxes):
                (x1, y1, x2, y2) = [int(v) for v in box]
                centroid = self.compute_centroid((x1, y1, x2, y2))
                # visualizing box
                cv2.rectangle(frame, (x1, y1), (x2, y2),
                              color, rect_thickness)
                # visualizing center of box
                cv2.circle(frame, (centroid[0], centroid[1]), circle_radius, color, circle_thickness)
        return frame, boxes

    def clear_tracker(self):
        self.multi_tracker = cv2.MultiTracker()

    @staticmethod
    def compute_centroid(bounding_box):
        (x1, y1, x2, y2) = bounding_box
        x_center = int((x2 - x1)/2 + x1)
        y_center = int((y2 - y1)/2 + y1)
        return (x_center, y_center)

    @staticmethod
    def add_centroids(frame, bounding_box_list, circle_radius=4, color=(0, 0, 255), circle_thickness=-1):
        centroid_list = []
        for bounding_box in bounding_box_list:
            centroid = DroneTracker.compute_centroid(bounding_box)
            centroid_list.append(centroid)
            # visualizing center of box
            cv2.circle(frame, (centroid[0], centroid[1]), circle_radius, color, circle_thickness)
        return frame, centroid_list

class DroneVision():
    # this is roughly the pixel area of papa's face when he's 1meter away from the drone camera
    # self.bbox_area_when_1_m = 1700
    # roughly my face size when 1m away
    bbox_area_when_1_m = 2900
    def __init__(self, min_confidence=.8, setup_eye_detector=True, setup_tracker=False, tracker_model="kcf"):
        print("loading dnn model and weights from disk")
        self.model_path = os.path.join(os.path.dirname(__file__), "opencv_models",
                                       "res10_300x300_ssd_iter_140000_fp16.caffemodel")
        self.weights_path = os.path.join(os.path.dirname(__file__), "opencv_models", "deploy.prototxt.txt")
        self.net = cv2.dnn.readNetFromCaffe(self.weights_path, self.model_path)
        self.min_confidence = min_confidence
        if setup_eye_detector:
            print("loading eye detector")
            self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_eye.xml')

        self.detector_success = []
        self.tracker_initialized = False
        self.past_bbox = []
        if setup_tracker:
            self.drone_tracker = DroneTracker(tracker_model)


    def find_face(self, frame, one_face=False, font=cv2.FONT_HERSHEY_SIMPLEX, color=(0, 0, 255), rect_thickness=2,
                  font_scale=.2, font_thickness=2):
        start_time = time.time()
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
        # loop over the detections
        loop_start_time = time.time()
        box_list = []
        max_confidence = 0.0
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
            bounding_box = tuple(box.astype("int"))
            box_list.append(bounding_box)
            # draw the bounding box of the face along with the associated
            # probability
            text = "{:.2f}%".format(confidence * 100)
            y = bounding_box[1] - 10 if bounding_box[1] - 10 > 10 else bounding_box[1] + 10
            cv2.rectangle(frame, (bounding_box[0], bounding_box[1]), (bounding_box[2], bounding_box[3]),
                          color, rect_thickness)
            cv2.putText(frame, text, (bounding_box[0], y),
                        font, font_scale, color, font_thickness)
            # if only want one face, skip rest of occurences where confidence is less than prior.
            # this assumes max confidence is first in the array
            if one_face:
                break
        logging.debug("finished loop in {} seconds".format(time.time() - loop_start_time))
        return frame, box_list

    def detect_and_track(self, frame, use_tracker=True, font=cv2.FONT_HERSHEY_SIMPLEX, color=(0, 0, 255),
                         rect_thickness=2, font_scale=.2, font_thickness=2, find_eyes=False):
        """A function to detect faces and eyes, and if don't detect but had detected in the past,
        apply a tracker to the last face if the detector faces"""
        # always prefer the detector as it is more accurate
        frame, bbox_list = self.find_face(frame, one_face=True, font=font, color=color,
                                                         rect_thickness=rect_thickness,
                                                         font_scale=font_scale, font_thickness=font_thickness)
        frame, centroid_list = DroneTracker.add_centroids(frame, bbox_list)
        # if found some faces, search those regions for eyes, else search all of frame
        if len(bbox_list) > 0 and find_eyes:
            frame, eye_list = self.find_eyes(frame, bbox_list)
        elif find_eyes:
            frame, eye_list = self.find_eyes(frame, [(0, 0, frame.shape[0], frame.shape[1])])
        if find_eyes:
            frame, eye_centroid_list = DroneTracker.add_centroids(frame, eye_list)
            bbox_list += eye_centroid_list
        if not use_tracker:
            return frame, bbox_list
        # if bbox list isn't empty, ie detector found some faces
        if len(bbox_list) >= 1:
            self.past_bbox = bbox_list
            self.detector_success.append(True)
            if self.tracker_initialized:
                self.tracker_initialized = False
                self.drone_tracker.clear_tracker()
            return frame, bbox_list
        else:
            self.detector_success.append(False)
        # if the detector fails, we have a past bounding box, and tracker isn't initialized, initialize it
        if not self.detector_success[-1] and len(self.past_bbox) >= 1 and not self.tracker_initialized:
            logging.debug("detector failed, initializing tracker")
            self.drone_tracker.initialize_tracker(frame, self.past_bbox)
            self.tracker_initialized = True
            frame, bbox_list = self.drone_tracker.track(frame)
            self.past_bbox = bbox_list
            return frame, bbox_list
        # if detector fails, we have a past bounding box, and tracker is initialized then use it
        if not self.detector_success[-1] and len(self.past_bbox) >= 1 and self.tracker_initialized:
            logging.debug("using tracker")
            frame, bbox_list = self.drone_tracker.track(frame)
            self.past_bbox = bbox_list
            return frame, bbox_list
        # if detector fails and we don't have a past bounding box return the frame
        if not self.detector_success[-1] and len(self.past_bbox) == 0:
            logging.debug("detector didn't find anything, no past bbox so can't use tracker")
            self.tracker_initialized = False
            return frame, bbox_list

    def find_eyes(self, frame, roi_list, color=(0, 255, 0), rect_thickness=2):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eye_locations = []
        for roi in roi_list:
            (x1, y1, x2, y2) = roi
            roi_gray = gray[y1:y1 + y2, x1:x1 + x2]
            # did trial and error with the minNeighbors--it seems to not like my left eye but 13 seems reasonable
            # to pick up the right one while avoiding false positives
            eyes = self.eye_cascade.detectMultiScale(roi_gray, minNeighbors=13)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(frame, (ex + x1, ey + y1), (ex + ew + x1, ey + eh + y1), (0, 127, 255), 2)
                eye_locations.append((ex + x1, ey + y1, ex + ew + x1, ey + eh + y1))
        return frame, eye_locations

    @staticmethod
    def calculate_distance(frame, bbox, font=cv2.FONT_HERSHEY_SIMPLEX, color=(0, 0, 255),
                           font_scale=.4, font_thickness=2):
        bbox_area = bbox[0] * bbox[1]

        distance = bbox_area / DroneVision.bbox_area_when_1_m
        distance = round(distance, 2)

        cv2.putText(frame, "Distance is {}m".format(distance), (0, frame.shape[0] - 20),
                    font, font_scale, color, font_thickness)

        return frame, distance
