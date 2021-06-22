from .state import State
import depthai as dai
import numpy as np

class ReidState(State):
    def __init__(self, device, manager):
        super().__init__(device, manager)
        self.state_name = "reid"
        self.reid_threshold = 50
        self.counter = 0
        self.sm = manager
        self.mc = self.sm.GetMotorController()

    def cos_dist(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def update(self):
        super().update()
        self.mc.stopMotors()

        self.counter += 1
        if self.counter > self.reid_threshold:
            for t in self.trackletsData:
                roi = t.roi.denormalize(self.frame.shape[1], self.frame.shape[0])
                x1 = int(roi.topLeft().x)
                y1 = int(roi.topLeft().y)
                x2 = int(roi.bottomRight().x)
                y2 = int(roi.bottomRight().y)

                # check which tracked person corresponds to our target person

                if self.statusMap[self.trackletsData[0].status] == "TRACKED":
                    det_frame = self.frame[y1:y2, x1:x2]
                    nn_data = dai.NNData()
                    nn_data.setLayer("data", self.to_planar(det_frame, (128, 256)))
                    self.device.getInputQueue("reid_in").send(nn_data)


            tracked = self.sm.GetTargetPerson()
            not_found = True
            for t in self.trackletsData:
                if len(self.trackletsData) > 0 and self.statusMap[self.trackletsData[0].status] == "TRACKED":
                    reid_result = self.device.getOutputQueue("reid_nn").get().getFirstLayerFp16()
                    dist = self.cos_dist(reid_result, tracked[0])
                    if dist > 0.7:
                        print("Found target person with tracking id : " + str(t.id))
                        print(dist)
                        self.sm.SetTargetPersonId(t.id)
                        not_found = False
                        break

            print(str(not_found))
            if not_found == False:
                # go to tracking mode
                self.sm.TargetReidentified()

            self.counter = 0
