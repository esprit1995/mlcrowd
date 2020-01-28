# make necessary imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data
from torch.optim import *
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np


##################################################
##################################################
##########    AE CLASS DEFINITION    ############
##################################################
##################################################

class AEv3(nn.Module):
    """
    parameterizable VAE module
    """

    def __init__(self,
                 dim_input,
                 dim_output,
                 hidden_activation="relu",
                 batch_norm=False):
        """
        flexible constructor for Neural network: allow for user-defined input
        and latent space dimensions
        :param dim_input: required, dimensionality of input data
        :param dim_output: required, dimensionality of latent space
        :param activation: which activation function to use.
                           Options: ["relu", "leaky_relu", "rrelu", "elu"]
        :param batch_norm: whether to apply batch normalization
        """
        super(AEv3, self).__init__()

        self.dim_input = dim_input
        self.dim_output = dim_output
        self.activation = hidden_activation
        self.batch_norm = batch_norm

        self.cv1 = nn.Sequential(nn.Conv1d(in_channels=1,
                                           out_channels=3,
                                           kernel_size=5,
                                           stride=1,
                                           padding=1),
                                 nn.BatchNorm1d(3),
                                 nn.ReLU())
        self.cv2 = nn.Sequential(nn.Conv1d(in_channels = 3,
                                           out_channels = 6,
                                           kernel_size = 5,
                                           stride=1,
                                           padding=1),
                                 nn.BatchNorm1d(6),
                                 nn.ReLU())
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(6 * (dim_input - 4), 400)
        self.bn1 = nn.BatchNorm1d(400)
        self.fc2 = nn.Linear(400, dim_output)
        self.fc3 = nn.Linear(dim_output, 400)
        self.bn3 = nn.BatchNorm1d(400)
        self.fc4 = nn.Linear(400, dim_input)

    def encode(self, X):
        """
        encodes the input image into two vectors: mean and variance
        :param X: input data in torch Tensor format
        :returns: mu and var
        """
        activation_func = getattr(F, self.activation)
        if self.batch_norm:
            # get 3 maps, each of size dim_input+2-5+1 = dim_input-2
            hidden1 = self.cv2(self.cv1(torch.reshape(X, (-1, 1, self.dim_input))))

            # flatten, get 3*(dim_input-2) features
            hidden1_flat = self.flat(hidden1)

            # feed 3*(dim_input-2) features to a fully connected layer with BN
            hidden2 = activation_func(self.bn1(self.fc1(hidden1_flat)))
        else:
            # all the same, but no batch normalization
            hidden1 = self.cv2(self.cv1(torch.reshape(X, (-1, 1, self.dim_input))))
            hidden1_flat = self.flat(hidden1)
            hidden2 = activation_func(self.fc1(hidden1_flat))
        return self.fc2(hidden2)

    def decode(self, z):
        """
        project a tensor from the latent space back into original coordinates
        :param z: tensor in the latent space to be decoded
        """
        activation_func = getattr(F, self.activation)
        if self.batch_norm:
            hidden3 = activation_func(self.bn3(self.fc3(z)))
        else:
            hidden3 = activation_func(self.fc3(z))
        out = self.fc4(hidden3)
        return out / out.norm(dim=1, keepdim=True, p=2)

    def forward(self, x):
        """
        forward function of VAE NN
        :param x: input image in torch Tensor format
        returns: x decoded from latent space along with mean and logvar tensors
        """
        latent = self.encode(x.view(-1, self.dim_input))
        return self.decode(latent)


##################################################
##################################################
##########   AE WRAPPER DEFINITION   ############
##################################################
##################################################

class AEv3wrapper:
    """
    Class to facilitate VAE usage
    """

    def __init__(self,
                 dim_input,
                 dim_output,
                 trainloader,
                 testloader,
                 hidden_activation="relu",
                 batch_norm=False,
                 early_stop_M=5):
        self.activation = hidden_activation
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.batch_norm = batch_norm
        self.model = AEv3(dim_input, dim_output, hidden_activation, batch_norm)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=0.001)
        self.trainloader = trainloader
        self.testloader = testloader
        self.loss_vals_per_epoch = list()
        self.early_stop_M = early_stop_M

    def loss_func(self, reconstructed_x, x):
        """
        defines a loss for our VAE.
        """

        reconstr_loss = F.mse_loss(reconstructed_x, x.view(-1, self.dim_input))
        return reconstr_loss

    def train_AE(self, epoch):
        # put model in train mode
        self.model.train()
        train_loss = 0
        for batch_idx, data in enumerate(self.trainloader):
            # zero all gradients
            self.optimizer.zero_grad()
            reconstructed_batch = self.model(data)
            loss = self.loss_func(reconstructed_batch, data)

            # do backprop
            loss.backward()

            train_loss += loss.item()
            self.optimizer.step()
        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(self.trainloader.dataset)))

    def test_AE(self, epoch):
        # put model into evaluation mode
        self.model.eval()
        test_loss = 0

        # deactivate autograd engine (backprop become unavailable, but that speeds up computations and
        # reduces memory usage; also, we don't update weights here, so backprop is not needed).
        with torch.no_grad():
            for i, data in enumerate(self.testloader):
                recon_batch = self.model(data)
                test_loss += self.loss_func(recon_batch, data).item()
        test_loss /= len(self.testloader.dataset)
        # crude early stopping
        self.loss_vals_per_epoch.append(test_loss)
        if len(self.loss_vals_per_epoch) > self.early_stop_M:
            old_loss = self.loss_vals_per_epoch[epoch - 5]
            now_loss = self.loss_vals_per_epoch[-1]
            if now_loss / old_loss > 0.999:
                return False
        print('====> Test set loss: {:.4f}'.format(test_loss))
        return True
