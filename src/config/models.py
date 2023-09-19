from torch import nn
from torchvision import models, datasets

ARGS = {
    "mnist": (1, 256, 10),
    "emnist": (1, 256, 62),
    "fmnist": (1, 256, 10),
    "cifar10": (3, 400, 10),
    "cifar100": (3, 400, 100),
}


class LeNet5(nn.Module):
    def __init__(self, dataset, colorized=False) -> None:
        channel_count = ARGS[dataset][0]
        if(colorized and channel_count == 1):
            channel_count = 3

        super(LeNet5, self).__init__()
        self.net = nn.Sequential(
            # nn.Conv2d(3, 1, 1),
            # nn.LeakyReLU(inplace=True), 
            nn.Conv2d(1, 6, 5),
            nn.LeakyReLU(inplace=True), 
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(ARGS[dataset][1], 120),
            # nn.Linear(5000, 448),
            nn.LeakyReLU(inplace=True),
            nn.Linear(120, 84),
            nn.LeakyReLU(inplace=True),
            nn.Linear(84, ARGS[dataset][2]),
        )

    def forward(self, x):
        res = self.net(x)
        return res


class Diginet(nn.Module):
    def __init__(self, dataset, colorized=False) -> None:
        channel_count = ARGS[dataset][0]
        if(colorized and channel_count == 1):
            channel_count = 3

        super(Diginet, self).__init__()
        self.net = nn.Sequential(
            # nn.Conv2d(3, 1, 1),
            # nn.LeakyReLU(inplace=True), 
            nn.BatchNorm2d(3),
            nn.Conv2d(channel_count, 18, 3, 2, groups=3),
            #
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(18),
            nn.Conv2d(18, 32, 3, 2),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, 2),
            nn.LeakyReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(256, 120),
            # nn.Linear(5000, 448),
            nn.LeakyReLU(inplace=True),
            nn.Linear(120, 84),
            nn.LeakyReLU(inplace=True),
            nn.Linear(84, ARGS[dataset][2]),
        )

    def forward(self, x):
        res = self.net(x)
        return res


class Resnet18(nn.Module):
    def __init__(self, dataset, colorized=False) -> None:
        channel_count = ARGS[dataset][0]
        if(colorized and channel_count == 1):
            channel_count = 3
        
        super(Resnet18, self).__init__()

        self.net = models.resnet18()
        num_last_ftrs = self.net.fc.in_features

        # for param in self.net.parameters():
        #     param.requires_grad = False

        self.net.fc = nn.Linear(num_last_ftrs, ARGS[dataset][2])

    def forward(self, x):
        res = self.net(x)
        return res