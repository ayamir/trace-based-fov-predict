import argparse


from model import *
from preprocessor import *
from torch.utils.data import TensorDataset, DataLoader

from preprocessor import HW
from preprocessor import PW
from preprocessor import FPS
from preprocessor import DOWNSAMPLE
from preprocessor import MODELS_PATH

import cmder
import mlflow.pytorch
import torch
import torcheck
import math
import matplotlib
import numpy as np

matplotlib.use("agg")
import matplotlib.pyplot as plt


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Dataset:
    def __init__(self):
        self.hw_unit = math.floor(HW * FPS / DOWNSAMPLE)
        self.pw_unit = math.floor(PW * FPS / DOWNSAMPLE)

    def read_dataset(self, data_path):
        data = pd.read_csv(data_path, header=None).values
        # hw * fps / downsample: 1 * 30 / 2 = 15
        boundary = self.hw_unit * 4
        view_data = data[:, 0:boundary]
        # pw * fps / downsample: 1 * 30 / 2 = 15
        view_label = data[:, boundary:]
        return view_data, view_label

    def datasetmaker(self, data_path):
        data, label = self.read_dataset(data_path)
        data, label = torch.from_numpy(data), torch.from_numpy(label)
        data = data.view(-1, self.hw_unit, 4).float()
        label = label.view(-1, self.pw_unit, 4).float()
        dataset = TensorDataset(data, label)
        datasetloader = DataLoader(dataset=dataset, batch_size=30, shuffle=True)
        return datasetloader


def train_loop(dataloader, loss_name, loss_fn, epochs, lr):
    cmder.warningOut(f"Train models with {loss_name}")
    model_dir = MODELS_PATH + loss_name + "/models_e" + str(epochs) + "/"

    model = VPLSTM(hid_size=64, layers=1, input_size=4, output_size=4)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    torcheck.register(optimizer)
    torcheck.add_module_inf_check(model)
    torcheck.add_module_nan_check(model)

    size = len(dataloader.dataset)

    if os.path.isdir(model_dir):
        cmder.runCmd(f"mv {model_dir} ./models/{loss_name}_bak")
    cmder.runCmd(f"mkdir -p {model_dir}")

    loss_values = []
    for epoch in range(epochs):
        cmder.infOut(f"epoch{epoch}")
        loss_sum = []
        for batch, (X, y) in enumerate(dataloader):
            # Compute prediction and loss
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            # zero grad
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum.append(loss.item())
            if batch % 200 == 0 and batch != 0:
                loss, current = loss.item(), batch * len(X)
                cmder.infOut(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                with open(model_dir + "loss.txt", "a+", encoding="utf-8") as f:
                    f.write(
                        "epoch"
                        + str(epoch)
                        + " "
                        + str(sum(loss_sum) / len(loss_sum))
                        + "\n"
                    )
        loss_values.append(sum(loss_sum) / len(loss_sum))
        mlflow.pytorch.save_model(model, model_dir + str(epoch))
    plt.plot(np.linspace(1, epochs, epochs).astype(int), loss_values)
    plt.savefig(model_dir + "loss.png")


def test_loop(dataloader, loss_name, loss_fn, epochs):
    cmder.infOut(f"Test model with {loss_name}")
    model_dir = (
        MODELS_PATH + loss_name + "/models_e" + str(epochs) + "/" + str(epochs - 1)
    )
    test_dir = MODELS_PATH + loss_name + "/test_e" + str(epochs) + "/"

    if not os.path.isdir(test_dir):
        cmder.runCmd(f"mkdir -p {test_dir}")

    loss_sum = 0
    size = len(dataloader.dataset)
    model = mlflow.pytorch.load_model(model_dir)

    model.eval()
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            loss_sum += loss
            if batch % 200 == 0 and batch != 0:
                loss, current = loss.item(), batch * len(X)
                cmder.infOut(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                with open(test_dir + loss_name + ".txt", "a+", encoding="utf-8") as f:
                    f.write(str(loss_sum.item() / batch) + "\n")


def comparison_of_loss(is_train, epochs, lr):
    train_path = MODELS_PATH + "train.csv"
    test_path = MODELS_PATH + "test.csv"

    loss_fn_dict = {
        "MAE": torch.nn.L1Loss(),
        "MSE": torch.nn.MSELoss(),
        "HuberLoss": torch.nn.SmoothL1Loss(),
    }

    dataset = Dataset()

    if is_train:
        # clean()
        train_dataloader = dataset.datasetmaker(train_path)
        for loss_name in loss_fn_dict:
            train_loop(train_dataloader, loss_name, loss_fn_dict[loss_name], epochs, lr)
    else:
        test_dataloader = dataset.datasetmaker(test_path)
        for loss_name in loss_fn_dict:
            test_loop(test_dataloader, loss_name, loss_fn_dict[loss_name], epochs)


def clean():
    cmder.runCmd(f"mv {MODELS_PATH}models/ ./models_bak")
    cmder.runCmd(f"mv {MODELS_PATH}test/ ./test_bak")
    _, res = cmder.runCmd(f"ls")
    cmder.infOut(res.replace("\n", " "))


if __name__ == "__main__":
    gpu_list = [0, 1]
    gpu_list_str = ",".join(map(str, gpu_list))
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu_list_str)

    parser = argparse.ArgumentParser(description="Train or Test models.")
    parser.add_argument(
        "--train",
        metavar="0",
        type=int,
        help="1 means train, 0 means test. Default value is 0.",
        default=0,
        nargs="?",
    )
    parser.add_argument(
        "--clean",
        metavar="0",
        type=int,
        help="1 means clean models and tests. Default value is 0.",
        default=0,
        nargs="?",
    )
    args = parser.parse_args()
    is_train = True if args.train == 1 else False
    is_clean = True if args.clean == 1 else False
    batch_size = 256
    lr = 1e-3
    epochs = 50

    comparison_of_loss(is_train, epochs, lr)
