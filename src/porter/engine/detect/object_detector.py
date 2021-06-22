import cv2
import numpy as np
from pathlib import Path


class ObjectDetector:
    def __init__(self, pipeline):
        self.status = 0
        self.pipeline = pipeline
        self.device = None
        self.q_nn = None
        self.detection_nn = None
        self.detections = []
        self.blobPath = None
        self.labelMap = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
        self.personsDetected = []

    def frameNorm(self, frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    def getDeviceQueue(self, device):
        self.device = device
        self.q_nn = device.getOutputQueue(name="detections", maxSize=4, blocking=False)

    def createNeuralNetwork(self, cam_rgb):
        self.blobPath = str((Path(__file__).parent.parent.parent / Path('models/mobilenet-ssd.blob')).resolve().absolute())

    def returnBlobPath(self):
        return self.blobPath

    def createLinkOut(self):
        self.xout_nn = self.pipeline.createXLinkOut()
        self.xout_nn.setStreamName("detections")

        return self.xout_nn

    def detectPersons(self, frame):
        self.personsDetected = []
        for detection in self.detections:
            if self.labelMap[detection.label] == "person":
                self.personsDetected.append(detection)
                bbox = self.frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                cv2.putText(frame, self.labelMap[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5,
                            255)

    def update(self, frame):
        in_nn = self.q_nn.tryGet()
        if in_nn is not None:
            self.detections = in_nn.detections
            self.detectPersons(frame)
            return frame
        else:
            return None

    def getDetections(self):
        return self.detections

    def getPersonDetections(self):
        return self.personsDetected