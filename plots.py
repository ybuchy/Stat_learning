import torch
from train import bernoulli_decoder
import numpy as np
from torch.distributions.normal import Normal
from matplotlib import pyplot as plt

torch.manual_seed(0)

mvn = Normal(0, 1)

for dataset in ["", "_fashion"]:
    for z_dim in [3, 10, 20]:
        # L plot
        l = torch.load("losses" + dataset + f"_z{z_dim}").detach().numpy()
        x = 100 * np.array(list(range(len(l))))
        plt.xscale("log")
        plt.ylabel(r"$\mathcal{L}$", fontsize=16)
        plt.xlabel("number of data points")
        plt.plot(x, l, label=r"$N_z = $" + f"{z_dim}")

    plt.grid()
    plt.legend()
    plt.savefig("L_plot" + dataset + ".png")
    plt.clf()

for dataset in ["", "_fashion"]:
    for z_dim in [3, 10, 20]:
        # sample plot
        state_dict = torch.load("decoder" + dataset + f"_z{z_dim}")
        decoder = bernoulli_decoder(z_dim, 500, 28 * 28)
        decoder.load_state_dict(state_dict)
        for k in range(100):
            z = mvn.icdf(torch.rand(z_dim))
            plt.subplot(10, 10, k+1)
            plt.imshow(torch.sigmoid(decoder(z)).reshape((28, 28)).detach().numpy(), cmap="gray_r")
            plt.axis("off")
            plt.margins(0)
        plt.savefig("mfd" + dataset + f"_z{z_dim}.png")
