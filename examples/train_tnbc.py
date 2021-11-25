# coding=utf-8
# Copyright (c) Andreas Panteli, Jonas Teuwen

"""Training example."""

import argparse
import logging
import random
import sys
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.backends import cudnn
from torch.utils.data import DataLoader, Dataset

from ece_loss.loss import ExclusiveCrossEntropyLoss


class TnbcConstants(NamedTuple):
    """Constants for the TNBC dataset"""

    img_extension: str = "png"
    mask_extension: str = "png"
    img_width: int = 512
    img_height: int = 512
    img_channels: int = 3
    num_classes: int = 2


def set_seed(seed):
    """Sets the integer seed for training"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        cudnn.enabled = True
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.cuda.manual_seed_all(seed)


class TNBC(Dataset):
    """
    Loads TNBC images

    Normalisation to range [-1, 1] is used for every input sample. It applies no augmentations or other performance
    enhancing operations to keep the implementation simple.
    """

    def __init__(self, data_path):
        self.constants = TnbcConstants()
        self.normaliser = self.img_normaliser()
        self.denormaliser = self.img_denormaliser()
        self.data = self.get_data(data_path)

    @staticmethod
    def img_path_to_numpy(im_path):
        """Loads image path using PIL image to an numpy array"""
        return np.array(Image.open(im_path))

    @staticmethod
    def scale_to_range_0_1(image):
        """Scales an torch tensor image in the range [0, 1]"""
        image = image - torch.min(image)
        return image / torch.max(image)

    @staticmethod
    def img_normaliser(max_range_value=1, max_range_scaling_factor=2, max_range_offset=-1):
        """Returns the normaliser function for the input data"""

        def _norm(data):
            return data / max_range_value * max_range_scaling_factor + max_range_offset

        return _norm

    @staticmethod
    def img_denormaliser(max_range_value=1, max_range_scaling_factor=2, max_range_offset=-1):
        """Returns the de-normaliser function for the input data"""

        def _denorm(data):
            return data - max_range_offset * max_range_value / max_range_scaling_factor

        return _denorm

    def load_image_to_numpy(self, img_path):
        """Loads an image path to a numpy array and returns the array up to the first img_channels"""
        img = self.img_path_to_numpy(img_path)
        return img[..., : self.constants.img_channels]

    def load_mask_to_numpy(self, mask_path):
        """Loads a mask path to a numpy array and makes the mask binary"""
        return (self.img_path_to_numpy(mask_path) > 0) * 1.0

    def __len__(self):
        """return the length of the samples"""
        return len(self.data)

    def __getitem__(self, index):
        """Gets the sample at index `index`"""
        return self.data[index]

    def get_data(self, data_path):
        """Loads all data from the TNBC directory tree into a list dictionaries per image path"""
        data = []
        slide_list = [s for s in list(data_path.iterdir()) if "Slide" in s.name]
        for slide_path in slide_list:
            gt_slide_path = data_path / slide_path.name.replace("Slide_", "GT_")
            img_paths = list(slide_path.glob(f"*.{self.constants.img_extension}"))
            for img_path in img_paths:
                mask_path = gt_slide_path / img_path.name.replace(
                    f".{self.constants.img_extension}", f".{self.constants.mask_extension}"
                )
                img = torch.FloatTensor(self.load_image_to_numpy(img_path))
                mask = torch.LongTensor(self.load_mask_to_numpy(mask_path))
                data.append(
                    {
                        "img_path": str(img_path),
                        "img": self.normaliser(self.scale_to_range_0_1(img)).permute(2, 0, 1),
                        "mask": mask,
                    }
                )
        return data


def get_loss(preds, target, model_dict):
    """Returns the average loss value"""
    loss = model_dict["loss"](preds, target)
    return loss.mean()


def get_segmentation_dice(preds, targets):
    """Estimates the DICE score"""
    preds_argmax = preds.argmax(dim=1)
    obj_mask = targets > 0
    dice_score = (2 * preds_argmax[obj_mask].sum()) / (preds_argmax.sum() + obj_mask.sum())
    return dice_score.cpu().numpy()


def propagate_loss(loss, model_dict):
    """Back-propagates the loss"""
    loss.backward()
    model_dict["opt"].step()


def run(model_dict, dataloader, device, mode):
    """Runs one epoch of training/evaluation through the dataloader"""
    dice = []
    losses = []
    for _, batch in enumerate(dataloader):
        imgs = batch["img"].to(device)
        targets = batch["mask"].to(device)
        if mode == "train":
            model_dict["opt"].zero_grad()
        preds = model_dict["model"](imgs)

        loss = get_loss(preds, targets, model_dict)
        if mode == "train":
            propagate_loss(loss, model_dict)

        losses.append(loss.detach().cpu().numpy())
        dice.append(get_segmentation_dice(preds.detach(), targets.detach()))
    return np.mean(dice), np.mean(losses)


def get_logger():
    """Returns the logger"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(f"{args.loss} log.txt")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt="%(asctime)s %(message)s", datefmt="%m/%d/%Y %I:%M:%S")
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def get_model_dict(arguments, device):
    """Creates the model, optimsier and loss functions and returns them in one dictionary"""
    if arguments.loss == "ce":
        loss = torch.nn.CrossEntropyLoss(reduction="none").to(device)
    elif arguments.loss == "ece":
        loss = ExclusiveCrossEntropyLoss().to(device)
    else:
        raise NotImplementedError

    model = torch.hub.load(
        "mateuszbuda/brain-segmentation-pytorch",
        "unet",
        in_channels=TnbcConstants().img_channels,
        out_channels=TnbcConstants().num_classes,
        init_features=32,
        pretrained=False,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=arguments.lr)

    return {"model": model, "opt": optimizer, "loss": loss}


def plot_graphs(train_losses, eval_losses, eval_dices):
    """Saves training graphs"""
    plt.plot(train_losses, color="r", label="train loss")
    plt.plot(eval_losses, color="b", label="eval loss")
    plt.xlabel("epochs")
    plt.ylabel(f"{args.loss} loss")
    plt.title(f"{args.loss} Loss graphs")
    plt.legend()
    plt.grid()
    plt.savefig(f"{args.loss} losses.png")
    plt.close()

    plt.plot(eval_dices, color="b", label="eval DICE")
    plt.xlabel("epochs")
    plt.ylabel("DICE")
    plt.title(f"{args.loss} DICE graphs")
    plt.legend()
    plt.grid()
    plt.savefig(f"{args.loss} DICE.png")
    plt.close()


def main(arguments):  # pylint: disable=R0914
    """Performs the experiment training"""
    logger = get_logger()
    logger.info(f"args: {arguments}\n")  # pylint: disable=W1203
    set_seed(arguments.seed)
    device = torch.device(arguments.device)

    train_dataloader = DataLoader(TNBC(Path(arguments.train_path)), batch_size=arguments.batch_size, pin_memory=True)
    eval_dataloader = DataLoader(TNBC(Path(arguments.eval_path)), batch_size=arguments.batch_size)
    test_dataloader = DataLoader(TNBC(Path(arguments.test_path)), batch_size=arguments.batch_size)

    model_dict = get_model_dict(arguments, device)

    max_eval_score = -1
    best_epoch = 0
    train_dices, train_losses = [], []
    eval_dices, eval_losses = [], []
    best_model = None
    try:
        for epoch in range(arguments.epochs):
            if arguments.loss == "ece":
                model_dict["loss"].set_epoch(epoch)

            model_dict["model"].train()
            train_dice, train_loss = run(model_dict, train_dataloader, device, mode="train")
            train_dices.append(train_dice)
            train_losses.append(train_loss)

            model_dict["model"].eval()
            with torch.no_grad():
                eval_dice, eval_loss = run(model_dict, eval_dataloader, device, mode="eval")
                eval_dices.append(eval_dice)
                eval_losses.append(eval_loss)

                if eval_dice > max_eval_score:
                    max_eval_score = eval_dice
                    if arguments.save_model:
                        torch.save(
                            {
                                "epoch": epoch,
                                "model_state_dict": model_dict["model"].state_dict(),
                                "optimizer_state_dict": model_dict["opt"].state_dict(),
                            },
                            f"{arguments.loss} best_model",
                        )
                    best_epoch = epoch
                    best_model = model_dict["model"].state_dict()

            logger.info(  # pylint: disable=W1203
                f"{epoch}/{arguments.epochs}. Loss: {train_loss:.3f}, {eval_loss:.3f}. "
                f"DICE: {train_dice:.3f}, {eval_dice:.3f}"
            )
            plot_graphs(train_losses, eval_losses, eval_dices)

    except KeyboardInterrupt:
        logger.info("\n\nProcess interrupted by user\n\n")

    model_dict["model"].load_state_dict(best_model)
    model_dict["model"].eval()
    with torch.no_grad():
        test_score, _ = run(model_dict, test_dataloader, device, mode="test")

    logger.info(f"\n---\n {arguments.loss} | Epoch {best_epoch}: {test_score} DICE\n---\n")  # pylint: disable=W1203


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exclusive cross-entropy example on TNBC")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument(
        "--loss",
        type=str,
        default="ece",
        help='Choose between cross-entropy ("ce") or ' 'exclusive cross-entropy ("ece")',
    )
    parser.add_argument("--train_path", type=str, default="tnbc_30/train")
    parser.add_argument("--eval_path", type=str, default="tnbc_30/eval")
    parser.add_argument("--test_path", type=str, default="tnbc_30/test")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("-d", dest="debug", action="store_true")
    parser.add_argument("-save_model", dest="save_model", action="store_true")
    args = parser.parse_args()

    if args.debug:
        args.epochs = 2
        args.batch_size = 2

    main(args)
