import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.distributions.multivariate_normal import MultivariateNormal
import torchvision.transforms as transforms
from matplotlib import pyplot as plt

class encoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size, mean_size):
        super().__init__()

        self.linear1 = torch.nn.Linear(input_size, hidden_size)
        self.activation = torch.nn.Tanh()
        self.linear2 = torch.nn.Linear(hidden_size, mean_size)
        self.linear3 = torch.nn.Linear(hidden_size, mean_size)

    def forward(self, x):
        x = self.linear1(x)
        h = self.activation(x)
        mu = self.linear2(h)
        log_var = self.linear3(h)
        return mu, log_var

class decoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size, mean_size):
        super().__init__()

        self.linear1 = torch.nn.Linear(input_size, hidden_size)
        self.activation = torch.nn.Tanh()
        self.linear2 = torch.nn.Linear(hidden_size, mean_size)
        self.linear3 = torch.nn.Linear(hidden_size, mean_size)

    def forward(self, x):
        x = self.linear1(x)
        h = self.activation(x)
        mu = self.linear2(h)
        log_var = self.linear3(h)
        return mu, log_var

class bernoulli_decoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size, mean_size):
        super().__init__()

        self.linear1 = torch.nn.Linear(input_size, hidden_size)
        self.activation = torch.nn.Tanh()
        self.linear2 = torch.nn.Linear(hidden_size, mean_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        y = self.linear2(x)

        return y

def loss_bernoulli(encoder_mean, encoder_logvar, decoder_y, x, epsilon):
    KL = 1/2 * torch.sum(torch.ones(encoder_mean.shape) + encoder_logvar - torch.pow(encoder_mean, 2) - torch.exp(encoder_logvar), dim=1)

    ones = torch.ones((x.shape))
    log_likelihood = -torch.nn.functional.binary_cross_entropy_with_logits(decoder_y, x, reduction="none").sum(dim=1)

    return -(KL + log_likelihood)

if __name__ == "__main__":
    torch.manual_seed(0)
    M = 100 # Batch size
    L = 1

    for dataset in ["", "_fashion"]:
        for z_dim in [3, 10, 20]:
            if dataset == "_fashion":
                mnist_train = datasets.FashionMNIST(root="data", train=True, download=True, transform=transforms.ToTensor())
            else:
                mnist_train = datasets.MNIST(root="data", train=True, download=True, transform=transforms.ToTensor())

            img_tensor, _ = mnist_train[0]

            x_dim = img_tensor.shape[1] * img_tensor.shape[2]
            num_hidden = 500

            encoder_model = encoder(x_dim, num_hidden, z_dim)
            decoder_model = bernoulli_decoder(z_dim, num_hidden, x_dim)

            # Initialize parameters
            for param in encoder_model.parameters():
                nn.init.normal_(param, mean=0.0, std=0.01)
            for param in decoder_model.parameters():
                nn.init.normal_(param, mean=0.0, std=0.01)

            train_dataloader = DataLoader(mnist_train, batch_size=M, shuffle=True)

            mvn = MultivariateNormal(torch.zeros((z_dim,)), torch.eye(z_dim))

            optimizer = torch.optim.Adam(list(encoder_model.parameters()) + list(decoder_model.parameters()), lr=1e-3)

            num_epochs = 200
            losses = torch.zeros((num_epochs * len(mnist_train) // M))
            for epoch in range(num_epochs):
                print(f"epoch: {epoch}")
                for k, (inp, _) in enumerate(train_dataloader):
                    inp = inp.flatten(start_dim = 1)
                    inp = (inp > 0.5).float()
                    optimizer.zero_grad()
                    enc_mean, enc_logvar = encoder_model(inp)
                    epsilon = torch.stack(tuple(mvn.sample() for _ in range(100)))
                    z = enc_mean + torch.exp(0.5 * enc_logvar) * epsilon
                    dec = decoder_model(z)
                    l = loss_bernoulli(enc_mean, enc_logvar, dec, inp.flatten(start_dim=1), epsilon).mean()
                    l.backward()
                    losses[epoch * len(mnist_train) // M + k] = l
                    optimizer.step()

            torch.save(encoder_model.state_dict(), "encoder" + dataset + f"_z{z_dim}")
            torch.save(decoder_model.state_dict(), "decoder" + dataset + f"_z{z_dim}")
            torch.save(-losses, "losses" + dataset + f"_z{z_dim}")

            plt.xscale("log")
            plt.plot(-losses.detach().numpy())
            plt.savefig("loss" + dataset + f"_z{z_dim}.png")
