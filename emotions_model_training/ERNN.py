from torch import nn

# Currently dummy network for checking if training and loaders are working properly

class ERNN(nn.Module):
    def __init__(self, block, layers, num_classes = 7):
        super(ERNN, self).__init__()

        self.inplanes = 64
        # A little larger kernel to test on slightly larger images
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.maxpool1 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

        self.resblock1 = self.make_layer(block, 64, layers[0], stride=1)
        self.resblock2 = self.make_layer(block, 128, layers[1], stride=2)
        self.resblock3 = self.make_layer(block, 256, layers[2], stride=2)
        self.resblock4 = self.make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.flat = nn.Flatten()
        self.fullcon = nn.Linear(2048, 7)
        self.softmax = nn.Softmax()


    def make_layer(self, block, planes, num_blocks, stride=1):
        resiudal_val = None
        if stride != 1 or self.inplanes != planes:
            resiudal_val = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, residual=resiudal_val))
        self.inplanes = planes
        for i in range(1, num_blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.avgpool(x)
        x = self.flat(x)
        x = self.softmax(self.fullcon(x))

        return x
