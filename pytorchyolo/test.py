#! /usr/bin/env python3

from __future__ import division

import argparse
import tqdm
import numpy as np

from terminaltables import AsciiTable

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

from pytorchyolo.models import load_model
from pytorchyolo.utils.utils import load_classes, ap_per_class, get_batch_statistics, non_max_suppression, to_cpu, xywh2xyxy, print_environment_info
from pytorchyolo.utils.datasets import ListDataset
from pytorchyolo.utils.transforms import DEFAULT_TRANSFORMS
from options.test_options import TestOptions
from top import Top

class Test(Top):

    def __init__(self):
        opt = TestOptions().parse()  # get test options
        self.run(opt)

    def init_parameters(self):
        pass

    def evaluate_model_file(self, model_path, weights_path, img_path, class_names, batch_size=8, img_size=416,
                            n_cpu=8, iou_thres=0.5, conf_thres=0.5, nms_thres=0.5, verbose=True):
        """Evaluate model on validation dataset.

        :param model_path: Path to model definition file (.cfg)
        :type model_path: str
        :param weights_path: Path to weights or checkpoint file (.weights or .pth)
        :type weights_path: str
        :param img_path: Path to file containing all paths to validation images.
        :type img_path: str
        :param class_names: List of class names
        :type class_names: [str]
        :param batch_size: Size of each image batch, defaults to 8
        :type batch_size: int, optional
        :param img_size: Size of each image dimension for yolo, defaults to 416
        :type img_size: int, optional
        :param n_cpu: Number of cpu threads to use during batch generation, defaults to 8
        :type n_cpu: int, optional
        :param iou_thres: IOU threshold required to qualify as detected, defaults to 0.5
        :type iou_thres: float, optional
        :param conf_thres: Object confidence threshold, defaults to 0.5
        :type conf_thres: float, optional
        :param nms_thres: IOU threshold for non-maximum suppression, defaults to 0.5
        :type nms_thres: float, optional
        :param verbose: If True, prints stats of model, defaults to True
        :type verbose: bool, optional
        :return: Returns precision, recall, AP, f1, ap_class
        """
        dataloader = self._create_data_loader(
            img_path, batch_size, img_size, n_cpu)
        model = load_model(model_path, weights_path)
        metrics_output = self._evaluate(
            model,
            dataloader,
            class_names,
            img_size,
            iou_thres,
            conf_thres,
            nms_thres,
            verbose)
        return metrics_output


    def print_eval_stats(self, metrics_output, class_names, verbose):
        if metrics_output is not None:
            precision, recall, AP, f1, ap_class = metrics_output
            if verbose:
                # Prints class AP and mean AP
                ap_table = [["Index", "Class", "AP"]]
                for i, c in enumerate(ap_class):
                    ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
                print(AsciiTable(ap_table).table)
            print(f"---- mAP {AP.mean():.5f} ----")
        else:
            print("---- mAP not measured (no detections found by model) ----")


    def _evaluate(self, model, dataloader, class_names, img_size, iou_thres, conf_thres, nms_thres, verbose):
        """Evaluate model on validation dataset.

        :param model: Model to evaluate
        :type model: models.Darknet
        :param dataloader: Dataloader provides the batches of images with targets
        :type dataloader: DataLoader
        :param class_names: List of class names
        :type class_names: [str]
        :param img_size: Size of each image dimension for yolo
        :type img_size: int
        :param iou_thres: IOU threshold required to qualify as detected
        :type iou_thres: float
        :param conf_thres: Object confidence threshold
        :type conf_thres: float
        :param nms_thres: IOU threshold for non-maximum suppression
        :type nms_thres: float
        :param verbose: If True, prints stats of model
        :type verbose: bool
        :return: Returns precision, recall, AP, f1, ap_class
        """
        model.eval()  # Set model to evaluation mode

        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        labels = []
        sample_metrics = []  # List of tuples (TP, confs, pred)
        for _, imgs, targets in tqdm.tqdm(dataloader, desc="Validating"):
            # Extract labels
            labels += targets[:, 1].tolist()
            # Rescale target
            targets[:, 2:] = xywh2xyxy(targets[:, 2:])
            targets[:, 2:] *= img_size

            imgs = Variable(imgs.type(Tensor), requires_grad=False)

            with torch.no_grad():
                outputs = model(imgs)
                outputs = non_max_suppression(outputs, conf_thres=conf_thres, iou_thres=nms_thres)

            sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

        if len(sample_metrics) == 0:  # No detections over whole validation set.
            print("---- No detections over whole validation set ----")
            return None

        # Concatenate sample statistics
        true_positives, pred_scores, pred_labels = [
            np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
        metrics_output = ap_per_class(
            true_positives, pred_scores, pred_labels, labels)

        self.print_eval_stats(metrics_output, class_names, verbose)

        return metrics_output


    def _create_data_loader(self, img_path, batch_size, img_size, n_cpu):
        """
        Creates a DataLoader for validation.

        :param img_path: Path to file containing all paths to validation images.
        :type img_path: str
        :param batch_size: Size of each image batch
        :type batch_size: int
        :param img_size: Size of each image dimension for yolo
        :type img_size: int
        :param n_cpu: Number of cpu threads to use during batch generation
        :type n_cpu: int
        :return: Returns DataLoader
        :rtype: DataLoader
        """
        dataset = ListDataset(img_path, img_size=img_size, multiscale=False, transform=DEFAULT_TRANSFORMS)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=n_cpu,
            pin_memory=True,
            collate_fn=dataset.collate_fn)
        return dataloader


    def run(self, parser):
        
        args = parser.parse_args()
        print(f"Command line arguments: {args}")

        # Load configuration from data file
        data_config = self.parse_data_config(args.data)
        # Path to file containing all images for validation
        valid_path = data_config["valid"]
        class_names = load_classes(data_config["names"])  # List of class names

        precision, recall, AP, f1, ap_class = self.evaluate_model_file(
            args.model,
            args.weights,
            valid_path,
            class_names,
            batch_size=args.batch_size,
            img_size=args.img_size,
            n_cpu=args.n_cpu,
            iou_thres=args.iou_thres,
            conf_thres=args.conf_thres,
            nms_thres=args.nms_thres,
            verbose=True)


if __name__ == "__main__":
    Test = Test()
