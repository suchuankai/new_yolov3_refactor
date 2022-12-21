from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument("-w", "--weights", type=str, default="weights/yolov3.weights", help="Path to weights or checkpoint file (.weights or .pth)")
        parser.add_argument("-d", "--data", type=str, default="config/coco.data", help="Path to data config file (.data)")
        parser.add_argument("-b", "--batch_size", type=int, default=8, help="Size of each image batch")
        parser.add_argument("-v", "--verbose", action='store_true', help="Makes the validation more verbose")
        parser.add_argument("--img_size", type=int, default=416, help="Size of each image dimension for yolo")
        parser.add_argument("--iou_thres", type=float, default=0.5, help="IOU threshold required to qualify as detected")
        return parser
