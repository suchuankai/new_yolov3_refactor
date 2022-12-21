from .base_options import BaseOptions


class DetectOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument("-w", "--weights", type=str, default="weights/yolov3.weights", help="Path to weights or checkpoint file (.weights or .pth)")
        parser.add_argument("-i", "--images", type=str, default="data/samples", help="Path to directory with images to inference")
        parser.add_argument("-c", "--classes", type=str, default="data/coco.names", help="Path to classes label file (.names)")
        parser.add_argument("-o", "--output", type=str, default="output", help="Path to output directory")
        parser.add_argument("-b", "--batch_size", type=int, default=1, help="Size of each image batch")
        parser.add_argument("--img_size", type=int, default=416, help="Size of each image dimension for yolo")

        return parser
