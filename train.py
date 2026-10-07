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
        #self.squash = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        y = self.linear2(x)
        #y = self.squash(x)
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
    KL = 1/2 * torch.sum(torch.ones(encoder_mean.shape) + encoder_logvar - torch.pow(encoder_mean, 2) - torch.exp(encoder_logvar), dim=1)

    ones = torch.ones((x.shape))
    # TODO AI USAGE: earlier implementation not numerically stable, using pytorch implementation now
    log_likelihood = -torch.nn.functional.binary_cross_entropy_with_logits(decoder_y, x, reduction="none").sum(dim=1)
    # log_likelihood = torch.sum(x * torch.log(decoder_y) + (ones - x) * torch.log(ones - decoder_y), dim=1)

    return -(KL + log_likelihood)

if __name__ == "__main__":
    torch.manual_seed(0)
    M = 100 # Batch size
    L = 1

    mnist_train = datasets.MNIST(root="data", train=True, transform=transforms.ToTensor())
    mnist_test = datasets.MNIST(root="data", train=False, transform=transforms.ToTensor())

    img_tensor, _ = mnist_train[0]

    x_dim = img_tensor.shape[1] * img_tensor.shape[2]
    num_hidden = 500
    z_dim = 20

    encoder_model = encoder(x_dim, num_hidden, z_dim)
    decoder_model = bernoulli_decoder(z_dim, num_hidden, x_dim)

    # Initialize parameters
    for param in encoder_model.parameters():
        #nn.init.normal_(param, mean=0.0, std=np.sqrt(0.01))
        nn.init.normal_(param, mean=0.0, std=0.01)
    for param in decoder_model.parameters():
        #nn.init.normal_(param, mean=0.0, std=np.sqrt(0.01))
        nn.init.normal_(param, mean=0.0, std=0.01)

    # TODO data set, random samples
    train_dataloader = DataLoader(mnist_train, batch_size=M, shuffle=True)

    mvn = MultivariateNormal(torch.zeros((z_dim,)), torch.eye(z_dim))

    # TODO global stepsize chosen from [0.01, 0.02, 0.1] based on performance
    # based on performance on training set in first few iterations
    # TODO weight decay????????????????????
    #optimizer = torch.optim.Adagrad(list(encoder_model.parameters()) + list(decoder_model.parameters()), lr=0.02, weight_decay=1e-4, initial_accumulator_value=1e-6,
    #eps=1e-10,
#)
    optimizer = torch.optim.Adam(list(encoder_model.parameters()) + list(decoder_model.parameters()), lr=1e-3)

    num_epochs = 200
    losses = torch.zeros((num_epochs * len(mnist_train) // M))
    # TODO how did they track losses?
    for epoch in range(num_epochs):
        print(f"epoch: {epoch}")
        for k, (inp, _) in enumerate(train_dataloader):
            inp = inp.flatten(start_dim = 1)
            # TODO this is testing
            inp = (inp > 0.5).float()
            #inp = torch.bernoulli(inp)
            optimizer.zero_grad()
            enc_mean, enc_logvar = encoder_model(inp)
            epsilon = torch.stack(tuple(mvn.sample() for _ in range(100)))
            z = enc_mean + torch.exp(0.5 * enc_logvar) * epsilon
            dec = decoder_model(z)
            l = loss_bernoulli(enc_mean, enc_logvar, dec, inp.flatten(start_dim=1), epsilon).mean()
            l.backward()
            losses[epoch * len(mnist_train) // M + k] = l
            optimizer.step()
            """
        for k, (inp, _) in enumerate(mnist_test):
            inp = inp.flatten()
            # TODO do this here???????????
            inp = torch.bernoulli(inp)
            enc_mean, enc_logvar = encoder_model(inp)
            dec = decoder_model(enc_mean)
            """

    torch.save(encoder_model.state_dict(), "encoder")
    torch.save(decoder_model.state_dict(), "decoder")
    torch.save(-losses, "losses")

    plt.xscale("log")
    plt.plot(-losses.detach().numpy())
    plt.savefig("loss.png")
