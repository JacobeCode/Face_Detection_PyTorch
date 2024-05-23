from torch import nn

# Classic Residual Block (maybe will test diffrent setup's)
# Two Conv2D Residual layers - with possibility of easy passing the modifications to kernel etc.
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = [3, 3], stride = [1, 1], padding = [1, 1], residual = None):
        super(ResBlock, self).__init__()
        self.out_channels = out_channels
        self.residual = residual

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size[0], stride[0], padding[0]),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size[1], stride[1], padding[1]),
            nn.BatchNorm2d(out_channels)
        )
        self.relu2 = nn.ReLU()
        
    def forward(self, x):
        residual_val = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.residual:
            residual_val = self.residual(x)
        out += residual_val
        out = self.relu2(out)
        return out