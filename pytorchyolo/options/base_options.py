import argparse

class BaseOptions():
   
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        # basic parameters
        parser.add_argument("--n_cpu", type=int, default=8, help="Number of cpu threads to use during batch generation")
        parser.add_argument("-m", "--model", type=str, default="config/yolov3.cfg", help="Path to model definition file (.cfg)")
        parser.add_argument("--conf_thres", type=float, default=0.5, help="Object confidence threshold")
        parser.add_argument("--nms_thres", type=float, default=0.4, help="IOU threshold for non-maximum suppression")
        self.initialized = True
        return parser

    def gather_options(self):
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        return parser


    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        self.opt = opt
        print("in parse")
        return self.opt
