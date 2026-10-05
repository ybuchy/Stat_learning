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
        self.squash = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        h = self.activation(x)
        mu = self.linear2(h)
        mu = self.squash(mu)
        log_var = self.linear3(h)
        return mu, log_var

class bernoulli_decoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size, mean_size):
        super().__init__()

        self.linear1 = torch.nn.Linear(input_size, hidden_size)
        self.activation = torch.nn.Tanh()
        self.linear2 = torch.nn.Linear(hidden_size, mean_size)
        self.squash = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        y = self.squash(x)
        return y

# TODO when to not use gradients for tensors?????
def loss(encoder_mean, encoder_logvar, decoder_mean, decoder_logvar, x, epsilon):
    KL = 1/2 * torch.sum(torch.ones(encoder_mean.shape)  + encoder_logvar - torch.pow(encoder_mean, 2) - torch.exp(encoder_logvar))

    decoder_var = torch.exp(decoder_logvar)
    sigma_det = torch.prod(decoder_var)
    precision = torch.diag(torch.pow(decoder_var, -1))
    # TODO doesn't care for optimization?
    #log_likelihood = torch.log(1 / ((2 * np.pi) ** (decoder_mean.shape[0] / 2) * sigma_det ** (1 / 2))) + (-1/2 * (x - decoder_mean).T @ precision @ (x - decoder_mean))
    log_likelihood = -torch.log(sigma_det ** (1 / 2)) + (-1/2 * (x - decoder_mean).T @ precision @ (x - decoder_mean))

    return -(KL - log_likelihood)

def loss_bernoulli(encoder_mean, encoder_logvar, decoder_y, x, epsilon):
    KL = 1/2 * torch.sum(torch.ones(encoder_mean.shape)  + encoder_logvar - torch.pow(encoder_mean, 2) - torch.exp(encoder_logvar))

    decoder_var = torch.exp(decoder_logvar)
    sigma_det = torch.prod(decoder_var)
    precision = torch.diag(torch.pow(decoder_var, -1))
    log_likelihood = torch.sum(x * torch.log(decoder_y) + (torch.ones((x.shape[0],)) - x) * torch.log(torch.ones((x.shape[0],)) - decoder_y))

    return -(KL - log_likelihood)

if __name__ == "__main__":
    M = 100 # Batch size
    L = 1

    mnist_train = datasets.MNIST(root="data", train=True, transform=transforms.ToTensor())
    mnist_test = datasets.MNIST(root="data", train=False, transform=transforms.ToTensor())

    img_tensor, _ = mnist_train[0]

    x_dim = img_tensor.shape[1] * img_tensor.shape[2]
    num_hidden = 3
    z_dim = 2

    encoder_model = encoder(x_dim, num_hidden, z_dim)
    decoder_model = decoder(z_dim, num_hidden, x_dim)

    # Initialize parameters
    for param in encoder_model.parameters():
        nn.init.normal_(param, mean=0.0, std=np.sqrt(0.01))
    for param in decoder_model.parameters():
        nn.init.normal_(param, mean=0.0, std=np.sqrt(0.01))

    # TODO data set, random samples
    mvn = MultivariateNormal(torch.zeros((x_dim,)), torch.eye(x_dim))

    # TODO global stepsize chosen from [0.01, 0.02, 0.1] based on performance
    # based on performance on training set in first few iterations
    optimizer = torch.optim.Adagrad(list(encoder_model.parameters()) + list(decoder_model.parameters()), lr=0.01)

    num_epochs = 1
    losses = torch.zeros((num_epochs * len(mnist_train)))
    # TODO how did they track losses?
    for epoch in range(num_epochs):
        for k, (input, _) in enumerate(mnist_train):
            if k % 1000 == 0:
                print(k)
            optimizer.zero_grad()
            enc = encoder_model(input.flatten())
            # #TODO Just put mean?
            dec = decoder_model(enc[0])
            epsilon = mvn.sample()
            # TODO x = input?
            l = loss(*enc, *dec, input.flatten(), epsilon)
            l.backward()
            losses[epoch * len(mnist_train) + k] = l
            optimizer.step()

    plt.plot(losses.detach().numpy())
    plt.savefig("loss2.png")
