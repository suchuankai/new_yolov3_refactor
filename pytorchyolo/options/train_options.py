from .base_options import BaseOptions


class TrainOptions(BaseOptions):

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument("-d", "--data", type=str, default="config/coco.data", help="Path to data config file (.data)")
        parser.add_argument("-e", "--epochs", type=int, default=300, help="Number of epochs")
        parser.add_argument("-v", "--verbose", action='store_true', help="Makes the training more verbose")
        parser.add_argument("--pretrained_weights", type=str, help="Path to checkpoint file (.weights or .pth). Starts training from checkpoint model")
        parser.add_argument("--checkpoint_interval", type=int, default=1, help="Interval of epochs between saving model weights")
        parser.add_argument("--evaluation_interval", type=int, default=1, help="Interval of epochs between evaluations on validation set")
        parser.add_argument("--multiscale_training", action="store_true", help="Allow multi-scale training")
        parser.add_argument("--iou_thres", type=float, default=0.5, help="Evaluation: IOU threshold required to qualify as detected")
        parser.add_argument("--logdir", type=str, default="logs", help="Directory for training log files (e.g. for TensorBoard)")
        parser.add_argument("--seed", type=int, default=-1, help="Makes results reproducable. Set -1 to disable.")
        return parser
