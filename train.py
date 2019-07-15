import argparse
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import imageio
import numpy as np
import random
from AutoEncoder import AutoEncoder
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

parser = argparse.ArgumentParser('AutoEncoder')
parser.add_argument('--N', default=5, type=int, help='learning rate in training')
parser.add_argument('--batch_size', type=int, default=128, help='batch size in training')
parser.add_argument('--epoch',  default=50, type=int, help='number of epoch in training')
parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
args = parser.parse_args()
N = args.N
DOWNLOAD_MNIST = False

# load data
transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        # torchvision.transforms.Normalize(0, 1),
    ]
)
train_data = torchvision.datasets.MNIST(
    root='/home/slcheng/AutoEncoder/mnist',
    train=True,
    transform=transform,
    download=DOWNLOAD_MNIST,
)
trian_loader = Data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)

# load model
auotencoder = AutoEncoder()
optimizer = torch.optim.Adam(auotencoder.parameters(), lr=args.learning_rate)
mse_loss = nn.MSELoss()


# traning process
def train():
    fig, ax = plt.subplots(2, N)
    ln, = plt.plot([], [], animated=True)

    # randomly sample N images to visualize
    train_num = train_data.data.shape[0]
    random.seed(train_num)
    slice = random.sample(list(range(0, train_num)), N)
    origin_image = torch.empty(N, 28*28)
    for i in range(N):
        origin_image[i] = train_data.train_data[slice[i]].view(-1, 28*28)\
                        .type(torch.FloatTensor)
        ax[0][i].imshow(np.reshape(origin_image.data.numpy()[i], (28, 28)))
        ax[0][i].set_xticks([])
        ax[0][i].set_yticks([])

    # begin to train
    images = []
    for epoch in range(args.epoch):
        for id, (x, label) in enumerate(trian_loader):
            b_x = x.view(-1, 28*28)
            b_y = x.view(-1, 28*28)

            encoded, decoded = auotencoder(b_x)
            loss = mse_loss(decoded, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if id % 100 == 0:
                print('Epoch: ', epoch, "| train loss: %.4f" % loss.data.item())
                _, decoded_data = auotencoder(origin_image)
                for i in range(N):
                    ax[1][i].clear()
                    ax[1][i].imshow(np.reshape(decoded_data.data.numpy()[i], (28, 28)), cmap='gray')
                    ax[1][i].set_xticks(())
                    ax[1][i].set_yticks(())
                plt.savefig("a.png")
                images.append(imageio.imread("a.png"))
    imageio.mimsave('gen.gif', images, duration=0.5)


if __name__ == '__main__':
    train()
