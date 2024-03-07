#
# This file includes the neural network classes, based on the PyTorch Module
#

from torch.nn import Conv2d, MaxPool2d, Linear, Flatten, Dropout, Module
import torch.nn.functional as F
import torch



class CNN(Module):
    """ Class for the normal CNN.
    """
    def __init__(self, dropout=0, channels=1):
        super().__init__()
        self.name = "cnn"
        self.conv_input = Conv2d(
            in_channels=channels, out_channels=32, kernel_size=(3, 3), padding="same"
        )
        self.conv_1 = Conv2d(
            in_channels=32, out_channels=32, kernel_size=(3, 3), padding="same"
        )
        self.conv_2 = Conv2d(
            in_channels=32, out_channels=64, kernel_size=(3, 3), padding="same"
        )
        self.conv_2_2 = Conv2d(
            in_channels=64, out_channels=64, kernel_size=(3, 3), padding="same"
        )
        self.conv_3 = Conv2d(
            in_channels=64, out_channels=128, kernel_size=(3, 3), padding="same"
        )        
        self.conv_3_2 = Conv2d(
            in_channels=128, out_channels=128, kernel_size=(3, 3), padding="same"
        )
        self.pool = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.flatten = Flatten()
        self.linear_1 = Linear(in_features=1152, out_features=64)
        self.linear_2 = Linear(in_features=64, out_features=32)
        self.output_1 = Linear(in_features=32, out_features=1)
        self.output_2 = Linear(in_features=32, out_features=1)
        self.dropout = Dropout(p=dropout)

    def forward(self, x):
        # First convolutions
        x = F.relu(self.conv_input(x))
        x = F.relu(self.conv_1(x))
        x = self.pool(x)
        x_res_in = F.relu(self.conv_2(x))
        x = x_res_in+F.relu(self.conv_2_2(x_res_in))
        x = self.pool(x)
        x_res_in = F.relu(self.conv_3(x))
        x = F.relu(self.conv_3_2(x_res_in))
        x = self.pool(x)


        # Linear layers
        x = self.flatten(x)
        x = F.relu(self.linear_1(x))
        x = self.dropout(x)
        x = F.relu(self.linear_2(x))
        x = self.dropout(x)
        output_1 = self.output_1(x)
        output_2 = F.sigmoid(self.output_2(x))
        output = torch.cat([output_1, output_2], dim=1)
        return torch.unsqueeze(output, dim = -1)
    


class CNN_ES(Module):
    """ Class for the energy network.
    """
    def __init__(self, dropout=0, channels=1, sample_dim = 100):
        super().__init__()
        self.name = "cnn_es"
        self.sample_dim = sample_dim
        self.conv_input = Conv2d(
            in_channels=channels, out_channels=32, kernel_size=(3, 3), padding="same"
        )
        self.conv_1 = Conv2d(
            in_channels=32, out_channels=32, kernel_size=(3, 3), padding="same"
        )
        self.conv_2 = Conv2d(
            in_channels=32, out_channels=64, kernel_size=(3, 3), padding="same"
        )
        self.conv_2_2 = Conv2d(
            in_channels=64, out_channels=64, kernel_size=(3, 3), padding="same"
        )
        self.conv_3 = Conv2d(
            in_channels=64, out_channels=128, kernel_size=(3, 3), padding="same"
        )        
        self.conv_3_2 = Conv2d(
            in_channels=128, out_channels=128, kernel_size=(3, 3), padding="same"
        )
        self.pool = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        #General layers
        self.flatten = Flatten()
        self.dropout = Dropout(p=dropout)
        self.batch_norm = torch.nn.BatchNorm1d(1152)

        #Layers for noise generation
        self.linear_noise_1 = Linear(in_features=1152, out_features=64)
        self.linear_noise_2 = Linear(in_features=64, out_features=32)

        #Output layers
        self.output_1 = Linear(in_features=32, out_features=1)
        self.output_2 = Linear(in_features=32, out_features=1)

    def forward(self, x):
        # First convolutions
        x = F.relu(self.conv_input(x))
        x = F.relu(self.conv_1(x))
        x = self.pool(x)
        x_res_in = F.relu(self.conv_2(x))
        x = x_res_in+F.relu(self.conv_2_2(x_res_in))
        x = self.pool(x)
        x_res_in = F.relu(self.conv_3(x))
        x = F.relu(self.conv_3_2(x_res_in))
        x = self.pool(x)
        x_split = self.flatten(x)

        # Create noise vector
        latent_dim = list(x_split.size())
        latent_dim.insert(1,self.sample_dim)
        noise = torch.randn(size=latent_dim, device=x_split.device) + 1
        x_noise = self.batch_norm(x_split)        
        x_noise = torch.unsqueeze(x_noise, dim = 1)
        x_noise = torch.mul(x_noise, noise)        
        x_noise = F.elu(self.linear_noise_1(x_noise))
        x_noise = self.dropout(x_noise)
        x_noise = F.elu(self.linear_noise_2(x_noise))
        x_noise = self.dropout(x_noise)

        #Create outputs
        output_1 = self.output_1(x_noise)
        output_2 = F.sigmoid(self.output_2(x_noise))
        output = torch.cat([output_1, output_2], dim=-1)
        output = torch.movedim(output, 1, 2)
        return output
    

class CNN_ES_theta(Module):
    """ Class for the CNN to train points of the extremal coefficiant function.
    """

    def __init__(self, dropout=0, channels=1, points=1, sample_dim=100):
        super().__init__()
        self.name = "cnn_es_theta"
        self.sample_dim = sample_dim
        self.conv_input = Conv2d(
            in_channels=channels, out_channels=32, kernel_size=(3, 3), padding="same"
        )
        self.conv_1 = Conv2d(
            in_channels=32, out_channels=32, kernel_size=(3, 3), padding="same"
        )
        self.conv_2 = Conv2d(
            in_channels=32, out_channels=64, kernel_size=(3, 3), padding="same"
        )
        self.conv_2_2 = Conv2d(
            in_channels=64, out_channels=64, kernel_size=(3, 3), padding="same"
        )
        self.conv_3 = Conv2d(
            in_channels=64, out_channels=128, kernel_size=(3, 3), padding="same"
        )
        self.conv_3_2 = Conv2d(
            in_channels=128, out_channels=128, kernel_size=(3, 3), padding="same"
        )
        self.pool = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # General layers
        self.flatten = Flatten()
        self.dropout = Dropout(p=dropout)
        self.batch_norm = torch.nn.BatchNorm1d(1152)

        # Layers for noise generation
        self.linear_noise_1 = Linear(in_features=1152, out_features=64)
        self.linear_noise_2 = Linear(in_features=64, out_features=32)

        # Output layers
        self.output_1 = Linear(in_features=32, out_features=points)  # number of points in h_points

    def forward(self, x):
        # First convolutions
        x = F.relu(self.conv_input(x))
        x = F.relu(self.conv_1(x))
        x = self.pool(x)
        x_res_in = F.relu(self.conv_2(x))
        x = x_res_in + F.relu(self.conv_2_2(x_res_in))
        x = self.pool(x)
        x_res_in = F.relu(self.conv_3(x))
        x = F.relu(self.conv_3_2(x_res_in))
        x = self.pool(x)
        x_split = self.flatten(x)

        # Create noise vector
        latent_dim = list(x_split.size())
        latent_dim.insert(1, self.sample_dim)
        noise = torch.randn(size=latent_dim, device=x_split.device) + 1
        x_noise = self.batch_norm(x_split)
        x_noise = torch.unsqueeze(x_noise, dim=1)
        x_noise = torch.mul(x_noise, noise)
        x_noise = F.elu(self.linear_noise_1(x_noise))
        x_noise = self.dropout(x_noise)
        x_noise = F.elu(self.linear_noise_2(x_noise))
        x_noise = self.dropout(x_noise)

        # Create outputs
        output_1 = F.sigmoid(self.output_1(x_noise)) + 1  # extremal coefficient function in (1,2)
        output = torch.movedim(output_1, 1, 2)
        return output
    

if __name__ == "__main__":
    net = CNN(channels=1)
    test = torch.rand(size=(32, 1, 30, 30))
    res = net(test)
    print(res.shape)

    pytorch_total_params = sum(p.numel() for p in net.parameters())
    print(pytorch_total_params)
