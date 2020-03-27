import torch


class CNN(torch.nn.Module):
    def __init__(self, device_to_use):
        super(CNN, self).__init__()
        self.device = device_to_use

        print("[i] Creating cnn layers")
        self.layer1 = torch.nn.Conv2d(4, 16, kernel_size=8, stride=4, padding=0)
        self.layer2 = torch.nn.ReLU()
        self.layer3 = torch.nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0)
        self.layer4 = torch.nn.ReLU()
        self.layer5 = torch.nn.Linear(9*9*32, 256)
        self.layer6 = torch.nn.ReLU()
        self.out_layer = torch.nn.Linear(256, 4)

        print("[i] Initializing layers weights with nn.init.kaiming_normal_")
        self.layer1.apply(self.init_weights)
        self.layer3.apply(self.init_weights)
        self.layer5.apply(self.init_weights)
        self.out_layer.apply(self.init_weights)

    def init_weights(self, tensor):
        torch.nn.init.kaiming_normal_(tensor.weight)

    def forward(self, x):
        x = torch.from_numpy(x).float().to(self.device)
        x = x/255 # Normalizes input
        x = self.layer2(self.layer1(x))  # Conv+ReLU
        x = self.layer4(self.layer3(x))  # Conv+ReLU
        x = x.view(-1, 9*9*32)  # Flattening
        x = self.layer6(self.layer5(x))  # FC+ReLU
        x = self.out_layer(x)  # Full connected
        return x
