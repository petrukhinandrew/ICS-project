import cv2
import dlib
import numpy as np
from mylib.centroidtracker import CentroidTracker
from mylib.trackableobject import TrackableObject
from mylib import config
from telemetry import EntryTelemetry, TelemetryWrapper, TelemetryType
from datetime import datetime
NET_CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]


class EntryTracker:
    def __init__(self, telemetry_queue) -> None:
        self.centroid_tracker = CentroidTracker(40, 50)
        self.trackers = []
        self.trackable_objects = {}
        self.net = cv2.dnn.readNetFromCaffe(
            "mobilenet_ssd/MobileNetSSD_deploy.prototxt", "mobilenet_ssd/MobileNetSSD_deploy.caffemodel")
        self.telemetry_queue = telemetry_queue

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

    def _put_telemetry(self, direction):
        telemetry = EntryTelemetry(datetime.now(), direction)
        self.telemetry_queue.put(TelemetryWrapper(
            TelemetryType.T_ENTRY, telemetry))

    def _register_entries(self, objects):
        for (object_id, centroid) in objects.items():
            obj = self.trackable_objects.get(object_id, None)

            if obj is None:
                obj = TrackableObject(object_id, centroid)
                self.trackable_objects[object_id] = obj
                continue

            y_axis = [c[1] for c in obj.centroids]
            direction = centroid[1] - np.mean(y_axis)
            obj.centroids.append(centroid)

            if not obj.counted:  # push into queue instead
                if direction < 0 and centroid[1] < self.frame_height // 2:
                    self._put_telemetry("UP")
                    obj.counted = True

                elif direction > 0 and centroid[1] > self.frame_height // 2:
                    self._put_telemetry("DOWN")
                    obj.counted = True

            self.trackable_objects[object_id] = obj

    def process(self, frame, refresh_trackers):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rects = []
        if refresh_trackers:
            self._refresh_trackers(frame, rgb)
        else:
            rects = self._update_trackers(rgb)

        objects = self.centroid_tracker.update(rects)
        self._register_entries(objects)
