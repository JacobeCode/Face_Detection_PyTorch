from torch import nn

# Currently dummy network for checking if training and loaders are working properly

class ERNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=(197, 197), stride=120)
        self.relu1 = nn.ReLU()

        self.flatten = nn.Flatten()

        self.lin1 = nn.Linear(289, 7)
        self.soft = nn.Softmax()

    def forward(self, x):
        hidden = None

        x = self.relu1(self.conv1(x))
        x = self.flatten(x)
        x = self.soft(self.lin1(x))
        
        return x
