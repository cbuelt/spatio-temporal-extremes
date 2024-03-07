#
# This file includes the dataloaders for the neural network.
#

import numpy as np
from torch.utils.data import Dataset, DataLoader
from utils.utils import transform_parameters, generate_support_points
import torchvision.transforms as T
from torchvision.transforms.functional import rotate
import random
import torch
from evaluation.metrics import extremal_coefficient

""" import os
import sys
sys.path.append(os.getcwd())
from utils import transform_parameters, generate_support_points
from evaluation.metrics import extremal_coefficient """




def get_train_val_loader(
    data_path: str, model: str, type: str, batch_size: int = 64, batch_size_val: int = 64, points: int = 2, load_ext_coef: bool = False
):
    """Returns the training and validation dataloader.

    Args:
        data_path (str): Path to data folder.
        model (str): Max-stable model.
        type (str): Type of neural network to train.
        batch_size (int, optional): Training batch size. Defaults to 64.
        batch_size_val (int, optional): Validation batch size. Defaults to 64.
        points (int, optional): Number of points/parameters. Defaults to 2.
        load_ext_coef(bool, optional): Whether to load extremal coefficient function. Defaults to False.

    Returns:
        _type_: Training and validation dataloader and datasets.
    """
    train_dataset = SpatialField(data_path=data_path, model=model, var="train", dim=points, type = type, load_ext_coef=load_ext_coef)
    val_dataset = SpatialField(data_path=data_path, model=model, var="val", dim = points, type = type, load_ext_coef=load_ext_coef)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=False)
    return train_loader, val_loader, train_dataset, val_dataset


def get_test_loader(data_path: str, model: str, type: str, batch_size=750, points = 2, load_ext_coef: bool = False):
    """Returns the test data dataloader.

    Args:
        data_path (str): Path to data folder.
        model (str): Max-stable model.
        type (str): Type of neural network to train.
        batch_size (int, optional): Testing batch size. Defaults to 750.
        points (int, optional): Number of points/parameters. Defaults to 2.
        load_ext_coef(bool, optional): Whether to load extremal coefficient function. Defaults to False.

    Returns:
        _type_: Test data dataloader and dataset.
    """
    test_dataset = SpatialField(data_path=data_path, model=model, var="test", dim = points, type = type)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader, test_dataset


class SpatialField(Dataset):
    """Dataset class for max-stable field with parameters"""

    def __init__(
        self,
        data_path: str,
        model: str,
        var: str,
        dim: int,
        type: int,
        load_ext_coef: bool = False,
    ):
        """Initialize the dataset.

        Args:
            data_path (str): Path to data folder.
            model (str): Max-stable model.
            var (str): Variant of dataset, either training or validation/testing.
            dim (int, optional): Number of points/parameters.
            type (str): Type of neural network to train.
            load_ext_coef(bool, optional): Whether to load extremal coefficient function. Defaults to False.
        """
        self.data_path = data_path
        self.var = var
        self.type = type
        self.model = model        
        self.dim = dim
        self.img_data = np.load(
            self.data_path + self.model + "_" + self.var + "_data.npy"
        )
        self.sample_size = self.img_data.shape[0]

        try:
            if(self.type == "energy" or self.type == "normal"):
                self.param_data = np.load(
                    self.data_path + self.model + "_" + self.var + "_params.npy"
                )
            if (self.type == "energy_theta"):
                if load_ext_coef:
                    # Load extremal coefficient function
                    self.param_data = np.load(self.data_path + self.model + "_" + self.var + "_ext_coef.npy"
                )
                else:
                    # Create points of extremal coefficient function
                    h_support = generate_support_points()
                    params = np.load(
                        self.data_path + self.model + "_" + self.var + "_params.npy"
                    )
                    self.param_data = extremal_coefficient(h_support, self.model, params[:,0], params[:,1])
        except:
            if (self.type == "energy" or self.type == "normal"):
                self.param_data = np.ones(shape=(self.sample_size, 2))
            else:
                self.param_data = np.ones(shape=(self.sample_size, dim))


    def __len__(self) -> int:
        """Get length of the dataset.

        Returns:
            int: Length.
        """
        return self.sample_size

    def __getitem__(self, idx: int):
        """Get item from the dataset.

        Args:
            idx (int): Index of the data.

        Returns:
            _type_: Image and parameters of the sample.
        """
        img = self.img_data[idx, :, :]
        param = self.param_data[idx].astype("float32")

        # Transform
        img_mean = img.mean()
        img_std = img.std()
        img = (img - img_mean) / img_std
        if(self.type == "energy" or self.type == "normal"):
            param = transform_parameters(param)

        # Expand dimension of image
        img = np.expand_dims(img, axis=0).astype("float32")
        # img = img.astype("float32")

        if self.var == "train":
            # Rotation of image
            img = torch.from_numpy(np.swapaxes(img, 0, 2))
            angle = random.choice([0, 0, 180])
            img = rotate(torch.swapaxes(img, 0, 2), angle=angle)
            # Vertical and horizontal flip
            hflipper = T.RandomHorizontalFlip(p=0.2)
            vflipper = T.RandomVerticalFlip(p=0.2)
            img = hflipper(img)
            img = vflipper(img)
        return img, param


if __name__ == "__main__":
    exp = "normal"
    type = "energy_theta"
    model = "brown"
    data_path = f"data/{exp}/data/"
    train_loader, val_loader, train_dataset, val_dataset = get_train_val_loader(
        data_path=data_path, model=model, type = type
    )
    for sample in val_loader:
        img, param = sample
        break
    print(img.shape)
    print(param.shape)
    print(val_dataset.__len__())
