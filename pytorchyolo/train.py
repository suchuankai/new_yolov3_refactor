#! /usr/bin/env python3

from __future__ import division

import os
import argparse
import tqdm

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from pytorchyolo.models import load_model

from pytorchyolo.utils.logger import Logger
from pytorchyolo.utils.utils import to_cpu, load_classes, print_environment_info, provide_determinism, worker_seed_set
from pytorchyolo.utils.datasets import ListDataset
from pytorchyolo.utils.augmentations import AUGMENTATION_TRANSFORMS
from pytorchyolo.utils.loss import compute_loss
from pytorchyolo.utils.transforms import DEFAULT_TRANSFORMS
from terminaltables import AsciiTable
from pytorchyolo.test import _evaluate
from torchsummary import summary

from top import Top
from options.train_options import TrainOptions

class Train(Top):

    def __init__(self):
        opt = TrainOptions().parse()  # get test options
        self.run(opt)

    def _create_data_loader(self, img_path, batch_size, img_size, n_cpu, multiscale_training=False, mode='train'):
        
        if(mode=='train'):
            dataset = ListDataset(
                img_path,
                img_size=img_size,
                multiscale=multiscale_training,
                transform=AUGMENTATION_TRANSFORMS)
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=n_cpu,
                pin_memory=True,
                collate_fn=dataset.collate_fn,
                worker_init_fn=worker_seed_set)

        elif(mode=='valid'):
            dataset = ListDataset(img_path, img_size=img_size, multiscale=False, transform=DEFAULT_TRANSFORMS)
            dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=n_cpu,
            pin_memory=True,
            collate_fn=dataset.collate_fn)

        else:
            print("mode error!")

        return dataloader

    def init_parameters(self):
        pass

    def Create_optimizer(self, model):
        params = [p for p in model.parameters() if p.requires_grad]
       
        if (model.hyperparams['optimizer'] in [None, "adam"]):
            optimizer = optim.Adam(
                params,
                lr=model.hyperparams['learning_rate'],
                weight_decay=model.hyperparams['decay'],
            )
        elif (model.hyperparams['optimizer'] == "sgd"):
            optimizer = optim.SGD(
                params,
                lr=model.hyperparams['learning_rate'],
                weight_decay=model.hyperparams['decay'],
                momentum=model.hyperparams['momentum'])
        else:
            print("Unknown optimizer. Please choose between (adam, sgd).")
        return optimizer

    


    def run(self,parser):
        args = parser.parse_args()

        print(f"Command line arguments: {args}")

        if args.seed != -1:
            provide_determinism(args.seed)

        logger = Logger(args.logdir)  # Tensorboard logger

        # Create output directories if missing
        os.makedirs("output", exist_ok=True)
        os.makedirs("checkpoints", exist_ok=True)

        # Get data configuration
        data_config = self.parse_data_config(args.data)
        train_path = data_config["train"]
        valid_path = data_config["valid"]
        class_names = load_classes(data_config["names"])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ############
        # Create model
        # ############
       
        model = load_model(args.model, args.pretrained_weights)

        # Print model
        if args.verbose:
            summary(model, input_size=(3, model.hyperparams['height'], model.hyperparams['height']))

        mini_batch_size = model.hyperparams['batch'] // model.hyperparams['subdivisions']

        # #################
        # Create Dataloader
        # #################
        
        # Load training dataloader
        dataloader = self._create_data_loader(
            train_path,
            mini_batch_size,
            model.hyperparams['height'],
            args.n_cpu,
            args.multiscale_training,
            mode='train')

        # Load validation dataloader
        validation_dataloader = self._create_data_loader(
            valid_path,
            mini_batch_size,
            model.hyperparams['height'],
            args.n_cpu,
            mode='valid')

        # ################
        # Create optimizer
        # ################
        optimizer = self.Create_optimizer(model)
        

        # skip epoch zero, because then the calculations for when to evaluate/checkpoint makes more intuitive sense
        # e.g. when you stop after 30 epochs and evaluate every 10 epochs then the evaluations happen after: 10,20,30
        # instead of: 0, 10, 20
      
        for epoch in range(1, args.epochs+1):

            print("\n---- Training Model ----")

            model.train()  # Set model to training mode
       
            for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc=f"Training Epoch {epoch}")):
                batches_done = len(dataloader) * epoch + batch_i
      
                imgs = imgs.to(device, non_blocking=True)
                targets = targets.to(device)

                outputs = model(imgs)

                loss, loss_components = compute_loss(outputs, targets, model)

                loss.backward()

                ###############
                # Run optimizer
                ###############

                if batches_done % model.hyperparams['subdivisions'] == 0:
                    # Adapt learning rate
                    # Get learning rate defined in cfg
                    lr = model.hyperparams['learning_rate']
                    if batches_done < model.hyperparams['burn_in']:
                        # Burn in
                        lr *= (batches_done / model.hyperparams['burn_in'])
                    else:
                        # Set and parse the learning rate to the steps defined in the cfg
                        for threshold, value in model.hyperparams['lr_steps']:
                            if batches_done > threshold:
                                lr *= value
                    # Log the learning rate
                    logger.scalar_summary("train/learning_rate", lr, batches_done)
                    # Set learning rate
                    for g in optimizer.param_groups:
                        g['lr'] = lr

                    # Run optimizer
                    optimizer.step()
                    # Reset gradients
                    optimizer.zero_grad()

                # ############
                # Log progress
                # ############
                if args.verbose:
                    print(AsciiTable(
                        [
                            ["Type", "Value"],
                            ["IoU loss", float(loss_components[0])],
                            ["Object loss", float(loss_components[1])],
                            ["Class loss", float(loss_components[2])],
                            ["Loss", float(loss_components[3])],
                            ["Batch loss", to_cpu(loss).item()],
                        ]).table)

                # Tensorboard logging
                tensorboard_log = [
                    ("train/iou_loss", float(loss_components[0])),
                    ("train/obj_loss", float(loss_components[1])),
                    ("train/class_loss", float(loss_components[2])),
                    ("train/loss", to_cpu(loss).item())]
                logger.list_of_scalars_summary(tensorboard_log, batches_done)

                model.seen += imgs.size(0)

            # #############
            # Save progress
            # #############

            # Save model to checkpoint file
            if epoch % args.checkpoint_interval == 0:
                checkpoint_path = f"checkpoints/yolov3_ckpt_{epoch}.pth"
                print(f"---- Saving checkpoint to: '{checkpoint_path}' ----")
                torch.save(model.state_dict(), checkpoint_path)

            # ########
            # Evaluate
            # ########

            if epoch % args.evaluation_interval == 0:
                print("\n---- Evaluating Model ----")
                # Evaluate the model on the validation set
                metrics_output = _evaluate(
                    model,
                    validation_dataloader,
                    class_names,
                    img_size=model.hyperparams['height'],
                    iou_thres=args.iou_thres,
                    conf_thres=args.conf_thres,
                    nms_thres=args.nms_thres,
                    verbose=args.verbose
                )

                if metrics_output is not None:
                    precision, recall, AP, f1, ap_class = metrics_output
                    evaluation_metrics = [
                        ("validation/precision", precision.mean()),
                        ("validation/recall", recall.mean()),
                        ("validation/mAP", AP.mean()),
                        ("validation/f1", f1.mean())]
                    logger.list_of_scalars_summary(evaluation_metrics, epoch)


    

    



if __name__ == '__main__':
    train = Train()