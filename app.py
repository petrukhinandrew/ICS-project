from imutils.video import VideoStream
from imutils.video import FPS
from mylib import config
from mylib.centroidtracker import CentroidTracker
from mylib.trackableobject import TrackableObject
import imutils
import numpy as np
from luma_device import LumaDevice
from queue import Queue
from threading import Thread
import cv2
import dlib
import time

NET_CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

LUMA_DELAY = 10

class App:
    def __init__(self) -> None:
        self.telemetry_queue = Queue()
        self.__setup_video()
        self.__setup_net()
        self.__setup_trackers()
        self.__setup_luma_device()

    def __setup_net(self):
        self.net = cv2.dnn.readNetFromCaffe(
            "mobilenet_ssd/MobileNetSSD_deploy.prototxt", "mobilenet_ssd/MobileNetSSD_deploy.caffemodel")

    def __setup_luma_device(self):
        self.luma_frame_queue = Queue()
        self.luma_device = LumaDevice(
            self.luma_frame_queue, self.telemetry_queue)
        self.luma_last_frame = time.time()
        self.luma_thread = Thread(target=self.luma_device.run, daemon=True)
        self.luma_thread.start()

    def __setup_video(self):
        self.vs = VideoStream(config.url).start()
        while self._get_frame() is None:
            continue
        self.frame_width, self.frame_height = self._get_frame().shape[:2]
        self.fps = FPS()
        self.total_frames = 0

    def __setup_trackers(self):
        self.centroid_tracker = CentroidTracker(40, 50)
        self.trackers = []
        self.trackable_objects = {}

    def _get_frame(self):
        frame = self.vs.read()
        return None if frame is None else imutils.resize(frame, width=500)

    def _refresh_trackers(self, frame, rgb):
        self.trackers.clear()

        blob = cv2.dnn.blobFromImage(
            frame, 0.007843, (self.frame_width, self.frame_height), 127.5)
        self.net.setInput(blob)
        detections = self.net.forward()

        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > config.Confidence:
                class_id = int(detections[0, 0, i, 1])
                if NET_CLASSES[class_id] != "person":
                    continue

                box = detections[0, 0, i, 3:7] * np.array(
                    [self.frame_width, self.frame_height, self.frame_width, self.frame_height])
                (startX, startY, endX, endY) = box.astype("int")

                corr_tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(startX, startY, endX, endY)
                corr_tracker.start_track(rgb, rect)

                self.trackers.append(corr_tracker)

    def _update_trackers(self, rgb):
        rects = []
        for tracker in self.trackers:
            tracker.update(rgb)
            pos = tracker.get_position()

            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())

            rects.append((startX, startY, endX, endY))
        return rects

    def _register_entries(self, objects):
        entry_saved = True
        for (object_id, centroid) in objects.items():
            obj = self.trackable_objects.get(object_id, None)

            if obj is None:
                obj = TrackableObject(object_id, centroid)
                self.trackable_objects[object_id] = obj
                continue

            y_axis = [c[1] for c in obj.centroids]
            direction = centroid[1] - np.mean(y_axis)
            obj.centroids.append(centroid)

            if not obj.counted: # push into queue instead
                if direction < 0 and centroid[1] < self.frame_height // 2:
                    totalUp += 1
                    obj.counted = True
                    entry_saved = False

                elif direction > 0 and centroid[1] > self.frame_height // 2:
                    totalDown += 1
                    obj.counted = True
                    entry_saved = False

            self.trackable_objects[object_id] = obj
        return entry_saved

    def _send_telemetry(self):
        while not self.telemetry_queue.empty():
            telemetry = self.telemetry_queue.get()
            print(telemetry)

    def run(self):
        while 1:
            frame = self._get_frame()
            cv2.imshow("lolkek", frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rects = []

            if time.time() - self.luma_last_frame > LUMA_DELAY:
                self.luma_frame_queue.put(frame.copy())
                self.luma_last_frame = time.time()

            if self.total_frames % config.SkipFrames == 0:
                self._refresh_trackers(frame, rgb)
            else:
                rects = self._update_trackers(rgb)

            objects = self.centroid_tracker.update(rects)
            self._register_entries(objects)
            
            self._send_telemetry()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = App()
    app.run()
