#!/usr/bin/env python3
import sys

sys.path.insert(0, '.')

import cv2
import depthai as dai
import numpy as np
import time
import math
from engine.track.Reidentifier import Reidentifier
from engine.spatial.spatial import Spatial
from engine.track.tracker import Tracker
from engine.stream.streamer import Streamer
from state.state_manager import StateManager
from engine.pose.pose_recognizer import PoseRecognizer


def frame_norm(frame, bbox):
    return (np.clip(np.array(bbox), 0, 1) * np.array([*frame.shape[:2], *frame.shape[:2]])[::-1]).astype(int)


def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
    resized = cv2.resize(arr, shape)
    return resized.transpose(2, 0, 1)


labelMap = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# Start defining a pipeline
pipeline = dai.Pipeline()

# Create the pipeline nodes
colorCamInstance = Streamer(pipeline)
colorCam = colorCamInstance.getColorCamera()
spatialInstance = Spatial(pipeline)
spatialDetectionNetwork = spatialInstance.getSpatialDetectionNetwork()
trackerInstance = Tracker(pipeline)
objectTracker = trackerInstance.initTrackerColorHistogram()
poseRecognizer = PoseRecognizer(pipeline)

# Create the links to send data to host
xoutRgb = pipeline.createXLinkOut()
trackerOut = pipeline.createXLinkOut()
xoutRgb.setStreamName("preview")
trackerOut.setStreamName("tracklets")

# Links between camera and neural network nodes
colorCam.preview.link(spatialDetectionNetwork.input)
objectTracker.passthroughTrackerFrame.link(xoutRgb.input)
objectTracker.out.link(trackerOut.input)
# Uncomment to get all detected objects
# objectTracker.passthroughDetections.link(create XLinkOut here to share to host)

spatialDetectionNetwork.passthrough.link(objectTracker.inputTrackerFrame)
spatialDetectionNetwork.passthrough.link(objectTracker.inputDetectionFrame)
spatialDetectionNetwork.out.link(objectTracker.inputDetections)

print("Creating Person Reidentification Neural Network...")
Reidentifier(pipeline)

trackedPeople = {}
next_id = 0
# Pipeline defined, now the device is connected to
with dai.Device(pipeline) as device:
    # Start the pipeline
    device.startPipeline()

    sm = StateManager(device)

    preview = device.getOutputQueue("preview", 4, False)
    tracklets = device.getOutputQueue("tracklets", 4, False)

    startTime = time.monotonic()
    counter = 0
    fps = 0
    frame = None
    reid_timer = 0
    reid_timer_threshold = 50
    color = (255, 0, 0)

    while (True):
        imgFrame = preview.get()
        track = tracklets.get()

        counter += 1

        current_time = time.monotonic()
        if (current_time - startTime) > 1:
            fps = counter / (current_time - startTime)
            counter = 0
            startTime = current_time

        frame = imgFrame.getCvFrame()
        trackletsData = track.tracklets

        # run the state manager updates
        sm.update(trackletsData, frame)

        for t in trackletsData:
            roi = t.roi.denormalize(frame.shape[1], frame.shape[0])
            x1 = int(roi.topLeft().x)
            y1 = int(roi.topLeft().y)
            x2 = int(roi.bottomRight().x)
            y2 = int(roi.bottomRight().y)
            label = labelMap[t.label]

            statusMap = {dai.Tracklet.TrackingStatus.NEW: "NEW", dai.Tracklet.TrackingStatus.TRACKED: "TRACKED",
                         dai.Tracklet.TrackingStatus.LOST: "LOST"}
            cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            cv2.putText(frame, f"ID: {[t.id]}", (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            cv2.putText(frame, statusMap[t.status], (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)

            cv2.putText(frame, f"X: {int(t.spatialCoordinates.x)} mm", (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_TRIPLEX,
                        0.5, color)
            cv2.putText(frame, f"Y: {int(t.spatialCoordinates.y)} mm", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX,
                        0.5, color)
            cv2.putText(frame, f"Z: {int(t.spatialCoordinates.z)} mm", (x1 + 10, y1 + 95), cv2.FONT_HERSHEY_TRIPLEX,
                        0.5, color)

        if reid_timer > reid_timer_threshold:
            reid_timer = 0

        cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)

        cv2.imshow("tracker", frame)

        if cv2.waitKey(1) == ord('q'):
            break
