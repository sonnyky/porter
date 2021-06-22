def loadTest():
    print("The capture module is loaded properly")

import cv2
import depthai as dai
import numpy as np

from src.engine.track.tracker import Tracker
from src.engine.stream.streamer import Streamer
from src.engine.detect.detector import Detector
from src.engine.detect.object_detector import ObjectDetector
from src.engine.spatial.spatial import Spatial
from src.engine.track.Reidentifier import Reidentifier

import queue
import signal
import threading

# Pipeline defined, now the device is connected to
def run():
    # Start defining a pipeline
    pipeline = dai.Pipeline()
    streamer = Streamer(pipeline)
    spatial = Spatial(pipeline)
    detector = ObjectDetector(pipeline)

    color_camera = streamer.createColorCamera()
    streamer.setParametersForSpatial()

    detector.createNeuralNetwork(color_camera)
    sn = spatial.setSpatialNetworkParameters(detector.returnBlobPath())

    xout_Rgb = streamer.createLinkOut(sn)
    xout_NN = detector.createLinkOut()
    spatial.createLinkOut(xout_NN, xout_Rgb)

    reidentifier = Reidentifier(pipeline)

    with dai.Device(pipeline) as device:
        # Start pipeline
        device.startPipeline()

        # Output queue will be used to get the rgb frames from the output defined above
        streamer.getDeviceQueue(device)
        detector.getDeviceQueue(device)
        spatial.getDeviceQueue(device)

        while True:
            in_rgb = streamer.update()

            # Retrieve 'bgr' (opencv format) frame
            if in_rgb is not None:
                frame = in_rgb.getCvFrame()
                drawnFrame = detector.update(frame)
                persons = detector.getPersonDetections()

                # get persons identification vectors
                #reidentifier.prepareFrame(frame)

                #if drawnFrame is not None:
                    #depthFrame = spatial.update(drawnFrame, detector.getDetections())
                    #cv2.imshow("depth", depthFrame)
                cv2.imshow("color", frame)

            if cv2.waitKey(1) == ord('q'):
                break
