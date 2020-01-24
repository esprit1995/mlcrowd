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

class AE(nn.Module):
    """
    parameterizable VAE module
    """

    def __init__(self, dim_input, dim_output):
        """
        flexible constructor for Neural network: allow for user-defined input
        and latent space dimensions
        :param dim_input: required, dimensionality of input data
        .param dim_output: required, dimensionality of latent space
        """
        super(AE, self).__init__()

        self.dim_input = dim_input
        self.dim_output = dim_output
        self.fc1 = nn.Linear(dim_input, 400)
        self.fc2 = nn.Linear(400, dim_output)
        self.fc3 = nn.Linear(dim_output, 400)
        self.fc4 = nn.Linear(400, dim_input)

    def encode(self, X):
        """
        encodes the input image into two vectors: mean and variance
        :param X: input data in torch Tensor format
        :returns: mu and var
        """
        hidden1 = F.relu(self.fc1(X))
        return self.fc2(hidden1)

    def decode(self, z):
        """
        project a tensor from the latent space back into original coordinates
        :param z: tensor in the latent space to be decoded
        """
        hidden3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(hidden3))

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

class AEwrapper:
    """
    Class to facilitate VAE usage
    """

    def __init__(self, dim_input, dim_output, trainloader, testloader):
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.model = AE(dim_input, dim_output)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=0.001)
        self.trainloader = trainloader
        self.testloader = testloader

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
        print('====> Test set loss: {:.4f}'.format(test_loss))
