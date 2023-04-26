from mylib.centroidtracker import CentroidTracker
from mylib.trackableobject import TrackableObject
from mylib.mailer import Mailer
from mylib import config, thread
import signal
from logger import Logger
from luma.luma_calculator import LumaCalculator

import numpy as np

from imutils.video import VideoStream
from imutils.video import FPS
import imutils
import time
import threading
import schedule

import dlib
import cv2

class RepeatTimer(threading.Timer):
    def run(self):
        while not self.finished.wait(self.interval):
            self.function(*self.args, **self.kwargs)


t0 = time.time()
logger = Logger()

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

    # initialize the frame dimensions (we'll set them as soon as we read
    # the first frame from the video)
    W = None
    H = None

    # instantiate our centroid tracker, then initialize a list to store
    # each of our dlib correlation trackers, followed by a dictionary to
    # map each unique object ID to a TrackableObject
    ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
    trackers = []
    trackableObjects = {}

    # initialize the total number of frames processed thus far, along
    # with the total number of objects that have moved either up or down
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
    # start the frames per second throughput estimator
    fps = FPS().start()
    luma_timer.start()
    if config.Thread:
        vs = thread.ThreadingClass(config.url)
    
    
    def exit_gracefully(sig, frame):
        fps.stop()
        luma_timer.cancel()
        print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

        if config.url is None:
            vs.stop()
        else:
            vs.release()

        # issue 15
        if config.Thread:
            vs.release()

        # close any open windows
        if config.ShowVideo:
            cv2.destroyAllWindows()
        exit(0)

    signal.signal(signal.SIGINT, exit_gracefully)

    # loop over frames from the video stream
    while True:
        # grab the next frame and handle if we are reading from either
        # VideoCapture or VideoStream
        frame = vs.read() if config.url is None else vs.read()[1]

        # resize the frame to have a maximum width of 500 pixels (the
        # less data we have, the faster we can process it), then convert
        # the frame from BGR to RGB for dlib
        frame = imutils.resize(frame, width=500)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # if the frame dimensions are empty, set them
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        # initialize the current status along with our list of bounding
        # box rectangles returned by either (1) our object detector or
        # (2) the correlation trackers
        status = "Waiting"
        rects = []

        # check to see if we should run a more computationally expensive
        # object detection method to aid our tracker
        if totalFrames % config.SkipFrames == 0:
            # set the status and initialize our new set of object trackers
            status = "Detecting"
            trackers = []

            # convert the frame to a blob and pass the blob through the
            # network and obtain the detections
            blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
            net.setInput(blob)
            detections = net.forward()

            # loop over the detections
            for i in np.arange(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated
                # with the prediction
                confidence = detections[0, 0, i, 2]

                # filter out weak detections by requiring a minimum
                # confidence
                if confidence > config.Confidence:
                    # extract the index of the class label from the
                    # detections list
                    idx = int(detections[0, 0, i, 1])

                    # if the class label is not a person, ignore it
                    if CLASSES[idx] != "person":
                        continue

                    # compute the (x, y)-coordinates of the bounding box
                    # for the object
                    box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                    (startX, startY, endX, endY) = box.astype("int")

                    # construct a dlib rectangle object from the bounding
                    # box coordinates and then start the dlib correlation
                    # tracker
                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(startX, startY, endX, endY)
                    tracker.start_track(rgb, rect)

                    # add the tracker to our list of trackers so we can
                    # utilize it during skip frames
                    trackers.append(tracker)

        # otherwise, we should utilize our object *trackers* rather than
        # object *detectors* to obtain a higher frame processing throughput
        else:
            # loop over the trackers
            for tracker in trackers:
                # set the status of our system to be 'tracking' rather
                # than 'waiting' or 'detecting'
                status = "Tracking"

                # update the tracker and grab the updated position
                tracker.update(rgb)
                pos = tracker.get_position()

                # unpack the position object
                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())

                # add the bounding box coordinates to the rectangles list
                rects.append((startX, startY, endX, endY))

        # draw a horizontal line in the center of the frame -- once an
        # object crosses this line we will determine whether they were
        # moving 'up' or 'down'
        if config.ShowVideo:
            cv2.line(frame, (0, H // 2), (W, H // 2), (0, 0, 0), 3)
            cv2.putText(frame, "-Prediction border - Entrance-", (10, H - ((i * 20) + 200)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # use the centroid tracker to associate the (1) old object
        # centroids with (2) the newly computed object centroids
        objects = ct.update(rects)

        # loop over the tracked objects
        for (objectID, centroid) in objects.items():
            # check to see if a trackable object exists for the current
            # object ID
            to = trackableObjects.get(objectID, None)

            # if there is no existing trackable object, create one
            if to is None:
                to = TrackableObject(objectID, centroid)

            # otherwise, there is a trackable object so we can utilize it
            # to determine direction
            else:
                # the difference between the y-coordinate of the *current*
                # centroid and the mean of *previous* centroids will tell
                # us in which direction the object is moving (negative for
                # 'up' and positive for 'down')
                y = [c[1] for c in to.centroids]
                direction = centroid[1] - np.mean(y)
                to.centroids.append(centroid)

                # check to see if the object has been counted or not
                if not to.counted:  # TODO() simply count frames when to process luma
                    # if the direction is negative (indicating the object
                    # is moving up) AND the centroid is above the center
                    # line, count the object
                    if direction < 0 and centroid[1] < H // 2:
                        totalUp += 1
                        to.counted = True
                        entry_saved = False

                    # if the direction is positive (indicating the object
                    # is moving down) AND the centroid is below the
                    # center line, count the object
                    elif direction > 0 and centroid[1] > H // 2:
                        totalDown += 1
                        to.counted = True
                        entry_saved = False

            # store the trackable object in our dictionary
            trackableObjects[objectID] = to

            # draw both the ID of the object and the centroid of the
            # object on the output frame
            if config.ShowVideo:
                text = "ID {}".format(objectID)
                cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.circle(
                    frame, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)

        # construct a tuple of information we will be displaying on the
        if config.ShowVideo:
            info = [
                ("Exit", totalUp),
                ("Enter", totalDown),
                ("Status", status),
            ]

            info2 = [
                ("Total people inside", totalDown - totalUp),
            ]

            # Display the output
            for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            for (i, (k, v)) in enumerate(info2):
                text = "{}: {}".format(k, v)
                cv2.putText(frame, text, (265, H - ((i * 20) + 60)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Initiate a simple log to save data at end of the day
        if config.Log:
            if luma_values != None:
                logger.log_luma(luma_values)
                luma_values = None
            if not entry_saved:
                logger.log_pass(totalDown, totalUp, totalDown - totalUp)
                entry_saved = True

        # show the output frame
        if config.ShowVideo:
            cv2.imshow("Real-Time Monitoring/Analysis Window", frame)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

        # increment the total number of frames processed thus far and
        # then update the FPS counter
        totalFrames += 1
        fps.update()

        if config.Timer:
            # Automatic timer to stop the live stream
            t1 = time.time()
            num_seconds = (t1-t0)
            if num_seconds > config.SchedulerTimeLimit:
                break

    # stop the timer and display FPS information
    fps.stop()
    luma_timer.cancel()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    vs.release()

    # issue 15
    if config.Thread:
        vs.release()

    # close any open windows
    if config.ShowVideo:
        cv2.destroyAllWindows()


# learn more about different schedules here: https://pypi.org/project/schedule/
if config.Scheduler:
    # Runs for every 1 second
    # schedule.every(1).seconds.do(run)
    # Runs at every day (09:00 am). You can change it.
    schedule.every().day.at(config.SchedulerStartTime).do(run)

    while 1:
        schedule.run_pending()

else:
    run()
