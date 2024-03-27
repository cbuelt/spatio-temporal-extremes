#
# This file includes all the code required for training the neural network.
#

import numpy as np
import torch
from torch import optim
import argparse
import os
import sys
sys.path.append(os.getcwd())
from utils.dataloader import get_train_val_loader, get_test_loader
from networks.models import CNN, CNN_ES, CNN_ES_theta
from utils.network import Scheduler
from utils.utils import retransform_parameters, generate_support_points
from utils.losses import EnergyScore

# Import config
from networks.config import config


def train_model(
    dir: str,
    model: str,
    epochs: int,
    batch_size: int,
    device,
    type: str = "normal",
    learning_rate: float = 0.0007,
    n_val: int = 500,
    dropout: float = 0.0,
    sample_dim: float = 100,
    points: int = 1,
    load_ext_coef: bool = False,
):
    """The function trains a model based on the given specification.

    Args:
        dir (str): The directory in the data folder, e.g. "normal", "application".
        model (str): The max-stable model to estimate.
        epochs (int): Number or training epochs.
        batch_size (int): Batch size.
        device (_type_): The CUDA device.
        type (str, optional): The network type (normal or energy). Defaults to "normal".
        learning_rate (float, optional): The learning rate. Defaults to 0.0007.
        n_val (int, optional): The Batch Size for the validation data. Defaults to 500.
        dropout (float, optional): Dropouit. Defaults to 0.0.
        sample_dim (float, optional): The number of samples to draw from the energy network. Defaults to 100.
        points (int, optional): The number of support points for the energy network. Defaults to 1.
        load_ext_coef(bool, optional): Whether to load extremal coefficient function. Defaults to False.

    Returns:
        _type_: Trained neural network.
    """
    # Set path
    path = f"data/{dir}/data/"
    # Get dataloaders
    train_dataloader, val_dataloader, _, _ = get_train_val_loader(
        data_path=path,
        model=model,
        batch_size=batch_size,
        batch_size_val=n_val,
        points=points,
        type=type,
        load_ext_coef=load_ext_coef,
    )
    # Specify model
    if type == "normal":
        net = CNN(dropout=dropout)
        loss_function = torch.nn.MSELoss()
    elif type == "energy":
        net = CNN_ES(sample_dim=sample_dim)
        loss_function = EnergyScore()
    elif type == "energy_theta":
        net = CNN_ES_theta(dropout=dropout, points=points, sample_dim=sample_dim)
        loss_function = EnergyScore()
    net.to(device)
    # Set optimizer
    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate)

    # Initialize Scheduler
    scheduler = Scheduler(
        path=f"data/{dir}/checkpoints/",
        name=f"{model}_{net.name}",
        patience=10,
        min_delta=0,
    )
    # Run experiment
    for epoch in range(1, epochs):
        train_loss = 0
        for sample in train_dataloader:
            img, param = sample
            img = img.to(device)
            param = param.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()
            net.train()

            # forward + backward + optimize
            outputs = net(img)
            param = torch.unsqueeze(param, -1)
            loss = loss_function(param, outputs)
            total_loss = loss
            total_loss.backward()
            optimizer.step()
            train_loss += total_loss.item() / len(train_dataloader)

        # Calculate val loss
        val_loss = 0
        for sample in val_dataloader:
            img, param = sample
            img = img.to(device)
            param = param.to(device)
            net.eval()
            outputs = net(img)
            param = torch.unsqueeze(param, -1)
            loss = loss_function(param, outputs)
            total_loss = loss
            val_loss += total_loss.item() / len(val_dataloader)
        print(
            f"Epoch: {epoch} \t Training loss: {train_loss:.4f} \t Validation loss: {val_loss:.4f}"
        )
        stop = scheduler(np.mean(val_loss), epoch, net)
        if stop:
            break
    return net


def predict_test_data(
    dir: str,
    model: str,
    device,
    test_size: int,
    type: str = "normal",
    sample_dim: int = 500,
    points: int = 0,
    load_ext_coef: bool = False,
):
    """This function calls a model checkpoint and predicts the test data.

    Args:
        dir (str): The directory in the data folder, e.g. "normal", "application".
        model (str): The max-stable model to estimate.
        device (_type_): The CUDA device.
        test_size (int): The size of the test dataset.
        type (str, optional): The network type (normal or energy). Defaults to "normal".
        batch_size_test (int, optional): The Batch size for the test data. Defaults to 500.
        sample_dim (float, optional): The number of samples to draw from the energy network. Defaults to 100.
        points (int, optional): The number of support points for the energy network. Defaults to 1.
        load_ext_coef(bool, optional): Whether to load extremal coefficient function. Defaults to False.
    """
    # Set path
    path = f"data/{dir}/data/"
    test_loader, _ = get_test_loader(
        data_path=path,
        model=model,
        type=type,
        batch_size=test_size,
        points=points,
        load_ext_coef=load_ext_coef,
    )

    # Load model
    if type == "normal":
        net = CNN()
    elif type == "energy":
        net = CNN_ES(sample_dim=sample_dim)
    elif type == "energy_theta":
        net = CNN_ES_theta(points=points, sample_dim=sample_dim)
    net.load_state_dict(torch.load(f"data/{dir}/checkpoints/{model}_{net.name}.pt"))

    # Send model to device
    net.to(device)

    # Prepare arrays
    if type == "normal":
        test_results = np.zeros(shape=(test_size, 2))
    elif type == "energy":
        test_results = np.zeros(shape=(test_size, 2, sample_dim))
    elif type == "energy_theta":
        test_results = np.zeros(shape=(test_size, points, sample_dim))

    # Calculate test samples in one batch
    for i, sample in enumerate(test_loader):
        img, param = sample
        img = img.to(device)
        param = param.to(device)
        net.eval()
        outputs = net(img)
        break
    if type == "normal":
        outputs = retransform_parameters(outputs.detach().cpu().numpy())
        test_results = (
            np.squeeze(outputs)
        )
    elif type == "energy":
        outputs = retransform_parameters(outputs.detach().cpu().numpy())
        test_results = (
            outputs
        )
    elif type == "energy_theta":
        outputs = outputs.detach().cpu().numpy()
        test_results = (
            np.squeeze(outputs)
        )

    np.save(file=f"data/{dir}/results/{model}_{net.name}.npy", arr=test_results)
    print(f"Saved results for model {model} and network {net.name}")


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Process some information.")
    parser.add_argument("--dir", type=str, help="Directory in the data folder.")
    parser.add_argument("--train", type=bool, default = False, help="Whether to train the model or evaluate only.")
    parser.add_argument("--load_ext_coef", type=bool, default = False, help="Whether to load extremal coefficient function or calculate it during training.")
    parser.add_argument("--model", type=str, help="Model to train.", default = None)
    args = parser.parse_args()

    train = args.train
    dir = args.dir
    load_ext_coef = args.load_ext_coef
    model = args.model

    # For reproducing results choose model
    if model is None:
        models = config["models_per_dir"][dir]
    else:
        models = [model]

    # Calculate support points
    h_support = generate_support_points(dh = config["dh"], max_length = config["max_length"])
    points = h_support.shape[0]
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for model in models:
        for type in config["network_types"]:
            # Train
            if train:
                trained_net = train_model(
                    dir,
                    model,
                    config["epochs"],
                    config["batch_size"],
                    device,
                    type=type,
                    points=points,
                    load_ext_coef=load_ext_coef,
                )
            # Predict
            predict_test_data(
                dir,
                model,
                device,
                test_size=config["test_size"],
                type=type,
                points=points,
                load_ext_coef=load_ext_coef,
            )
