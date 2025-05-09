import torch
import torch.nn as nn
import torch.nn.functional as F

class TabCNN(nn.Module):

    def __init__(
        self,
        dropout=0.25,
        spec_size=128,
        use_final_layer=True
    ) -> None:

        super().__init__()

        self.dropout = dropout
        self.use_final_layer = use_final_layer

        # Convolutional Layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3)).double() # Change input to float if this is too memory intensive
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3)).double()
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3)).double()
        
        # Max Pooling Layer
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        
        # Dense Layers
        # TODO: find a way to not hard code 5952
        features = ((spec_size - 6) / 2) * 64
        self.fc1 = nn.Linear(in_features=int(((spec_size - 6) / 2) * 64), out_features=256).double()  # size_after_conv_and_pool needs to be calculated based on input size
        #self.fc2 = nn.Linear(in_features=128, out_features=126).double()  # 126 = 6 * 21
        self.fc2 = nn.Linear(in_features=256, out_features=132).double() # 132 = 6 * 22

    def forward(self, x: torch.Tensor) -> torch.Tensor:

         # Applying convolutions, ReLU activations
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Max pool layer
        x = self.pool(x)

        # Dropout layer
        x = F.dropout(x, p=self.dropout)

        # Flattening
        x = torch.flatten(x, 1)
        
        # Dense layers with ReLU activation for the first
        x = F.relu(self.fc1(x))

        x = F.dropout(x, p=0.5)
        x = self.fc2(x)

        # Reshape to 6 x 21 and apply softmax activation to each row
        x = x.view(-1, 6, 22)
        if self.use_final_layer:
            x = F.softmax(x, dim=2)
        
        return x


if __name__ == "__main__":
    _ = TabCNN()