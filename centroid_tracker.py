from collections import OrderedDict

from scipy.spatial import distance as dist
import numpy as np


class CentroidTracker:
    def __init__(self, disappeared_remove_delay=50, max_centroid_dist=50):
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()

        self.disappeared_remove_delay = disappeared_remove_delay
        self.max_centroid_dist = max_centroid_dist

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        if len(rects) == 0:
            self.__update_disappeared()
            return self.objects

        input_centroids = np.zeros((len(rects), 2), dtype="int")

        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            input_centroids[i] = (cX, cY)

        if len(self.objects) == 0:
            self.__register_centroids(input_centroids)
        else:
            self.__update_objects(input_centroids)

        return self.objects

    def __update_disappeared(self):
        for objectID in list(self.disappeared.keys()):
            self.disappeared[objectID] += 1
            if self.disappeared[objectID] > self.disappeared_remove_delay:
                self.deregister(objectID)

    def __update_objects(self, input_centroids):
        objects_ids = list(self.objects.keys())
        objects_centroids = list(self.objects.values())
        centroids_pairwise_dists = dist.cdist(
            np.array(objects_centroids), input_centroids)

        rows = centroids_pairwise_dists.min(axis=1).argsort()
        cols = centroids_pairwise_dists.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        for (row, col) in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if centroids_pairwise_dists[row, col] > self.max_centroid_dist:
                continue

            object_id = objects_ids[row]
            self.objects[object_id] = input_centroids[col]
            self.disappeared[object_id] = 0

            used_rows.add(row)
            used_cols.add(col)

        unused_rows = set(
            range(0, centroids_pairwise_dists.shape[0])).difference(used_rows)
        unused_cols = set(
            range(0, centroids_pairwise_dists.shape[1])).difference(used_cols)

        if centroids_pairwise_dists.shape[0] >= centroids_pairwise_dists.shape[1]:
            for row in unused_rows:
                object_id = objects_ids[row]
                self.disappeared[object_id] += 1

                if self.disappeared[object_id] > self.disappeared_remove_delay:
                    self.deregister(object_id)
        else:
            for col in unused_cols:
                self.register(input_centroids[col])

    def __register_centroids(self, input_centroids):
        for i in range(0, len(input_centroids)):
            self.register(input_centroids[i])
