from torch import from_numpy
from torch.nn import Conv2d, Linear, init, Module
from torch.nn import functional as F


class CNN(Module):
    def __init__(self, device_to_use):
        super(CNN, self).__init__()
        self.device = device_to_use

        print("[i] Creating cnn layers")
        # Input: 4x84x84 tensor
        self.layer1 = Conv2d(4, 16, kernel_size=8, stride=4, padding=0)
        # Output layer1: 16x20x20 tensor
        self.layer3 = Conv2d(16, 32, kernel_size=4, stride=2, padding=0)
        # Output layer3: 32x9x9 tensor
        self.layer5 = Linear(9*9*32, 256)
        # Output layer5: 256
        self.out_layer = Linear(256, 4)
        # Output out_layer: 4

        print("[i] Initializing layers weights with nn.init.kaiming_normal_")
        self.layer1.apply(self.init_weights)
        self.layer3.apply(self.init_weights)
        self.layer5.apply(self.init_weights)
        self.out_layer.apply(self.init_weights)

    def init_weights(self, tensor):
        init.kaiming_normal_(tensor.weight)

    def forward(self, x):
        # Converting observation to tensor
        x = from_numpy(x).float().to('cpu')
        # INPUT TENSORS MUST BE PASSED AS DEPTHxHEIGHTxWIDTH
        # Input size: 4x84x84
        #print("Input to the cnn: ", x.size())
        x = x.unsqueeze(0) # Adding the batch dimension: 1x4x84x84
        x = x.permute(0, 3, 2, 1)
        #print("Input with batch_dimension ", x.size())
        x = x/255 # Normalizing input
        x = F.relu(self.layer1(x))  # Conv+ReLU. Input: 1x4x84x84. Output: 1x16x20x20
        #print("Conv1 + Relu output size: ", x.size())
        x = F.relu(self.layer3(x))  # Conv+ReLU  Input: 1x16x20x20. Output: 1x32x9x9
        #print("Conv2 + Relu output size: ", x.size())
        x = x.view(-1, 9*9*32)  # Flattening Input:  1x32x9x9. Output: 1x9*9*32
        #print("Flattening output size: ", x.size())
        x = x.squeeze(0) # Input:  1x32x9x9. Output: 9*9*32
        #print("Flattening output size without batch_dimension: ", x.size())
        x = F.relu(self.layer5(x))  # FC+ReLU Input: 9*9*32. Output: 256
        #print("Full connected + Relu output size: ", x.size())
        x = self.out_layer(x)  # Full connected. Input: 256. Output: 4
        #print("Out_layer output size: ", x.size())
        return x
