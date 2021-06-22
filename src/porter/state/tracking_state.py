from .state import State
import depthai as dai
import numpy as np


class TrackingState(State):
    def __init__(self, device, manager):
        super().__init__(device, manager)
        self.state_name = "tracking"

        self.lost_track_threshold = 20
        self.lost_track_counter = 0
        self.sm = manager
        self.mc = self.sm.GetMotorController()

    def cos_dist(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def update(self):
        super().update()
        trackingId = self.sm.GetTargetPersonId()

        if len(self.trackletsData) == 0:
            self.lost_track_counter += 1

        for t in self.trackletsData:
            if t.id != trackingId:
                continue
            elif t.id == trackingId and t.status == dai.Tracklet.TrackingStatus.TRACKED:
                # move motor to move robot to target
                self.lost_track_counter = 0
                self.mc.directionControl(t.spatialCoordinates.x, t.spatialCoordinates.y, t.spatialCoordinates.z)
            elif t.id == trackingId and t.status == dai.Tracklet.TrackingStatus.LOST:
                self.lost_track_counter += 1

        if self.lost_track_counter > self.lost_track_threshold:
            print("target lost, go to reid state")
            self.sm.TargetLost()
