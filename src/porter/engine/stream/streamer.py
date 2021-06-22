import depthai as dai


class Streamer:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.colorCam = pipeline.createColorCamera()
        self.colorCam.setPreviewSize(300, 300)
        self.colorCam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        self.colorCam.setInterleaved(False)
        self.colorCam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

    def getColorCamera(self):
        return self.colorCam

