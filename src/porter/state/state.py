import depthai as dai
import numpy as np
import cv2

class State:
    def __init__(self, device, manager):
        self.previous_state = None
        self.state_name = "base"
        self.device = device
        self.trackletsData = []
        self.frame = None

        self.statusMap = {dai.Tracklet.TrackingStatus.NEW: "NEW", dai.Tracklet.TrackingStatus.TRACKED: "TRACKED",
                 dai.Tracklet.TrackingStatus.LOST: "LOST"}
        self.sm = manager

    def updateData(self, trackletsData, frame):
        self.trackletsData = trackletsData
        self.frame = frame

    def update(self):
        pass

    def to_planar(self, arr: np.ndarray, shape: tuple) -> np.ndarray:
        resized = cv2.resize(arr, shape)
        return resized.transpose(2, 0, 1)
