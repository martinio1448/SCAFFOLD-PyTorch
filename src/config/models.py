from torch import nn

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
            nn.Conv2d(ARGS[dataset][0], 6, 5),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(ARGS[dataset][1], 120),
            nn.ReLU(True),
            nn.Linear(120, 84),
            nn.ReLU(True),
            nn.Linear(84, ARGS[dataset][2]),
        )

    def forward(self, x):
        return self.net(x)
