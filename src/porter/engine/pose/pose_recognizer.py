from pathlib import Path


class PoseRecognizer:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.pd_nn = pipeline.createNeuralNetwork()
        self.pd_nn.setBlobPath(str(Path("src/porter/models/pose/pose_detection.blob").resolve().absolute()))

        # create landmark model here too for now
        self.lm_nn = pipeline.createNeuralNetwork()
        self.lm_nn.setBlobPath(str(Path("src/porter/models/pose/pose_landmark_lite.blob").resolve().absolute()))
        self.lm_nn.setNumInferenceThreads(1)

        self.pd_in = pipeline.createXLinkIn()
        self.pd_in.setStreamName("pd_in")
        self.pd_in.out.link(self.pd_nn.input)

        # Pose detection output
        self.pd_out = pipeline.createXLinkOut()
        self.pd_out.setStreamName("pd_out")
        self.pd_nn.out.link(self.pd_out.input)

        self.lm_input_length = 256
        self.lm_in = pipeline.createXLinkIn()
        self.lm_in.setStreamName("lm_in")
        self.lm_in.out.link(self.lm_nn.input)
        # Landmark output
        self.lm_out = pipeline.createXLinkOut()
        self.lm_out.setStreamName("lm_out")
        self.lm_nn.out.link(self.lm_out.input)
