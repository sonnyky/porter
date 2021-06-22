import cv2
import numpy as np
from pathlib import Path


class Detector:
    def __init__(self, pipeline):
        self.status = 0
        self.pipeline = pipeline
        self.device = None
        self.q_nn = None
        self.detection_nn = None

    def frame_norm(self, frame, bbox):
        return (np.array(bbox) * np.array([*frame.shape[:2], *frame.shape[:2]])[::-1]).astype(int)

    def getDeviceQueue(self, device):
        self.device = device
        self.q_nn = device.getOutputQueue("nn")

    def createNeuralNetwork(self, cam_rgb):
        self.detection_nn = self.pipeline.createNeuralNetwork()
        self.detection_nn.setBlobPath(
            str((Path(__file__).parent.parent.parent / Path('models/person-detection-retail-0013.blob')).resolve().absolute()))
        cam_rgb.preview.link(self.detection_nn.input)

    def createLinkOut(self):
        self.xout_nn = self.pipeline.createXLinkOut()
        self.xout_nn.setStreamName("nn")
        self.detection_nn.out.link(self.xout_nn.input)

    def update(self, frame):
        in_nn = self.q_nn.tryGet()
        if in_nn is not None:
            # when data from nn is received, it is also represented as a 1D array initially, just like rgb frame
            bboxes = np.array(in_nn.getFirstLayerFp16())
            # the nn detections array is a fixed-size (and very long) array. The actual data from nn is available from the
            # beginning of an array, and is finished with -1 value, after which the array is filled with 0
            # We need to crop the array so that only the data from nn are left
            if bboxes.size != 0:
                bboxes = bboxes[:np.where(bboxes == -1)[0][0]]
                # next, the single NN results consists of 7 values: id, label, confidence, x_min, y_min, x_max, y_max
                # that's why we reshape the array from 1D into 2D array - where each row is a nn result with 7 columns
                bboxes = bboxes.reshape((bboxes.size // 7, 7))
                # Finally, we want only these results, which confidence (ranged <0..1>) is greater than 0.8, and we are only
                # interested in bounding boxes (so last 4 columns)
                bboxes = bboxes[bboxes[:, 2] > 0.8][:, 3:7]
                for raw_bbox in bboxes:
                    # for each bounding box, we first normalize it to match the frame size
                    bbox = self.frame_norm(frame, raw_bbox)
                    # and then draw a rectangle on the frame to show the actual result
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
                return frame
            else:
                return None
