import depthai as dai
from pathlib import Path

class Spatial:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.monoLeft = pipeline.createMonoCamera()
        self.monoRight = pipeline.createMonoCamera()
        self.spatialDetectionNetwork = pipeline.createMobileNetSpatialDetectionNetwork()
        self.stereo = pipeline.createStereoDepth()
        self.xoutBoundingBoxDepthMapping = None
        self.xoutDepth = None
        self.monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        self.monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
        self.monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        self.monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        self.depthQueue = None

        self.spatialDetectionNetwork.setBlobPath(
            str(Path("models/mobilenet-ssd.blob").resolve().absolute()))

        self.stereo.setOutputDepth(True)
        self.stereo.setConfidenceThreshold(255)
        self.spatialDetectionNetwork.setConfidenceThreshold(0.5)
        self.spatialDetectionNetwork.input.setBlocking(False)
        self.spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
        self.spatialDetectionNetwork.setDepthLowerThreshold(100)
        self.spatialDetectionNetwork.setDepthUpperThreshold(5000)

        # Create inputs and outputs

        self.monoLeft.out.link(self.stereo.left)
        self.monoRight.out.link(self.stereo.right)
        self.stereo.depth.link(self.spatialDetectionNetwork.inputDepth)

    def getSpatialDetectionNetwork(self):
        return self.spatialDetectionNetwork

