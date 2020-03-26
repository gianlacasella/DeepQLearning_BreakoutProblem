import torch


class CNN(torch.nn.module):
    def __init__(self, device_to_use):
        super(CNN, self).__init__()
        self.device = device_to_use
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(4, 16, kernel_size=8, stride=4, padding=0),
            torch.nn.ReLU()
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0),
            torch.nn.ReLU()
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Linear(9*9*32, 256),
            torch.nn.ReLU()
        )
        self.out_layer = torch.nn.Linear(256, 4)

    def forward(self, x):
        x = torch.from_numpy(x).float().to(self.device)
        x = self.layer1(x)  # Conv+ReLU
        x = self.layer2(x)  # Conv+ReLU
        x = x.view(-1, 9*9*32)  # Flattening
        x = self.layer3(x)  # FC+ReLU
        x = self.out_layer(x)  # Full connected
        return x
