from datetime import datetime

import cv2
import dlib
import numpy as np

from centroid_tracker import CentroidTracker
from trackable_object import TrackableObject
from telemetry import EntryTelemetry, TelemetryWrapper, TelemetryType

NET_CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]


class EntryTracker:
    def __init__(self, frame_width, frame_height, confidence, telemetry_queue) -> None:
        self.centroid_tracker = CentroidTracker(40, 50)
        self.trackers = []
        self.trackable_objects = {}

        self.net = cv2.dnn.readNetFromCaffe(
            "mobilenet_ssd/MobileNetSSD_deploy.prototxt", "mobilenet_ssd/MobileNetSSD_deploy.caffemodel")

        self.frame_width = frame_width
        self.frame_height = frame_height
        self.confidence = confidence
        self.telemetry_queue = telemetry_queue

    def __refresh_trackers(self, frame, rgb):
        self.trackers.clear()

        blob = cv2.dnn.blobFromImage(
            frame, 0.007843, (self.frame_width, self.frame_height), 127.5)
        self.net.setInput(blob)
        detections = self.net.forward()

        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > self.confidence:
                class_id = int(detections[0, 0, i, 1])
                if NET_CLASSES[class_id] != "person":
                    continue

                box = detections[0, 0, i, 3:7] * np.array(
                    [self.frame_width, self.frame_height, self.frame_width, self.frame_height])
                (x0, y0, x1, y1) = box.astype("int")

                corr_tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(x0, y0, x1, y1)
                corr_tracker.start_track(rgb, rect)

                self.trackers.append(corr_tracker)

    def __update_trackers(self, rgb):
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

    def __put_telemetry(self, direction):
        telemetry = EntryTelemetry(
            datetime.now().isoformat(sep=" "), direction)
        self.telemetry_queue.put(TelemetryWrapper(
            TelemetryType.T_ENTRY, telemetry))

    def __register_entries(self, frame, objects):
        coords = []
        for (object_id, centroid) in objects.items():
            obj = self.trackable_objects.get(object_id, None)

            if obj is None:
                obj = TrackableObject(object_id, centroid)
                self.trackable_objects[object_id] = obj
                coords.append((centroid[0], centroid[1]))
                continue

            mov_dir = centroid[1] - np.mean([c[1] for c in obj.centroids])
            obj.centroids.append(centroid)

            if not obj.counted:
                if mov_dir < 0 and centroid[1] < self.frame_height // 2:
                    self.__put_telemetry("UP")
                    print("UP")
                    obj.counted = True

                elif mov_dir > 0 and centroid[1] > self.frame_height // 2:
                    self.__put_telemetry("DOWN")
                    print("DOWN")
                    obj.counted = True
            coords.append((centroid[0], centroid[1]))
            self.trackable_objects[object_id] = obj
        return coords
    def process(self, frame, refresh_trackers):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rects = []
        if refresh_trackers:
            self.__refresh_trackers(frame, rgb)
        else:
            rects = self.__update_trackers(rgb)

        objects = self.centroid_tracker.update(rects)
        return  self.__register_entries(frame, objects)
