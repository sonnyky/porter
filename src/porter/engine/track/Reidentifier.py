from pathlib import Path

class Reidentifier:
    def __init__(self, pipeline):
        self.pipeline = pipeline

        self.reid_in = pipeline.createXLinkIn()
        self.reid_in.setStreamName("reid_in")
        self.reid_nn = pipeline.createNeuralNetwork()
        self.reid_nn.setBlobPath(
            str(Path("src/models/person-reidentification-retail-0277.blob").resolve().absolute()))

        # Decrease threads for reidentification
        self.reid_nn.setNumInferenceThreads(1)

        self.reid_nn_xout = pipeline.createXLinkOut()
        self.reid_nn_xout.setStreamName("reid_nn")
        self.reid_in.out.link(self.reid_nn.input)
        self.reid_nn.out.link(self.reid_nn_xout.input)

