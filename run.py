from mylib.centroidtracker import CentroidTracker
from mylib.trackableobject import TrackableObject
from mylib import config, thread
import signal
from logger import Logger
from luma.luma_calculator import LumaCalculator, PicLumaValues

import numpy as np

from imutils.video import VideoStream
from imutils.video import FPS
import imutils
import time
import threading

import dlib
import cv2
import datetime
from dataclasses import dataclass

@dataclass
class CameraTelemetry:
    timestamp: datetime.datetime = datetime.datetime.now()
    people_entered: int = 0
    people_left: int = 0
    luma_values: PicLumaValues = None

class RepeatTimer(threading.Timer):
    def run(self):
        while not self.finished.wait(self.interval):
            self.function(*self.args, **self.kwargs)


t0 = time.time()
logger = Logger()
telemetry = CameraTelemetry()

luma_values = None
frame = None


def run():
    global frame, luma_values

    # initialize the list of class labels MobileNet SSD was trained to
    # detect
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]

    # load our serialized model from disk
    net = cv2.dnn.readNetFromCaffe(
        "mobilenet_ssd/MobileNetSSD_deploy.prototxt", "mobilenet_ssd/MobileNetSSD_deploy.caffemodel")

    # run webcam stream if no url provided
    vs = VideoStream(config.url).start(
    ) if config.url != None else VideoStream().start()

    # put path to video file
    # vs = cv2.VideoCapture(videopath)
    time.sleep(2.0)

    W = None
    H = None

    ct = CentroidTracker(40, 50)
    trackers = []
    trackableObjects = {}

    totalFrames = 0
    totalDown = 0
    totalUp = 0
    entry_saved = True

    def calculate_luma() -> None:
        global frame
        global luma_values
        if frame is not None:
            luma_values = LumaCalculator.calculate(frame.copy())

    luma_timer = RepeatTimer(60, calculate_luma)
    
    fps = FPS().start()
    luma_timer.start()
    if config.Thread:
        vs = thread.ThreadingClass(config.url)
    
    
    def exit_gracefully():
        fps.stop()
        luma_timer.cancel()
        print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

        if config.url is None:
            vs.stop()
        else:
            vs.release()

        if config.Thread:
            vs.release()

        if config.ShowVideo:
            cv2.destroyAllWindows()
        exit(0)

    signal.signal(signal.SIGINT, exit_gracefully)

    def get_frame():
        return imutils.resize(vs.read() if config.url is None else vs.read()[1], width=500)

    while True:
        frame = get_frame()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if W is None or H is None:
            (H, W) = frame.shape[:2]

        rects = []

        if totalFrames % config.SkipFrames == 0:
            trackers = []

            blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
            net.setInput(blob)
            detections = net.forward()

            # loop over the detections
            for i in np.arange(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                if confidence > config.Confidence:
                    idx = int(detections[0, 0, i, 1])
                    if CLASSES[idx] != "person":
                        continue

                    box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                    (startX, startY, endX, endY) = box.astype("int")

                    corr_tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(startX, startY, endX, endY)
                    corr_tracker.start_track(rgb, rect)

                    trackers.append(corr_tracker)
        else:
            for tracker in trackers:
                tracker.update(rgb)
                pos = tracker.get_position()

                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())

                rects.append((startX, startY, endX, endY))

        objects = ct.update(rects)

        # loop over the tracked objects
        for (object_id, centroid) in objects.items():

            obj = trackableObjects.get(object_id, None)

            if obj is None:
                obj = TrackableObject(object_id, centroid)
                trackableObjects[object_id] = obj
                continue

            y_axis = [c[1] for c in obj.centroids]
            direction = centroid[1] - np.mean(y_axis)
            obj.centroids.append(centroid)

            if not obj.counted:
                if direction < 0 and centroid[1] < H // 2:
                    totalUp += 1
                    obj.counted = True
                    entry_saved = False

                elif direction > 0 and centroid[1] > H // 2:
                    totalDown += 1
                    obj.counted = True
                    entry_saved = False

            trackableObjects[object_id] = obj

        if config.Log:
            if luma_values != None:
                logger.log_luma(luma_values)
                luma_values = None
            if not entry_saved:
                logger.log_pass(totalDown, totalUp, totalDown - totalUp)
                entry_saved = True

        
        totalFrames += 1
        fps.update()

    exit_gracefully()
    
if __name__ == "__main__":
    run()
