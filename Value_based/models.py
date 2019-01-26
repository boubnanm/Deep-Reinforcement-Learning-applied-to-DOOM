import torch
import torch.nn as nn
import torch.nn.functional as F


def conv2d_size_out(size, kernel_size = 5, stride = 2):
    """
    Description
    --------------
    Compute the output dimension when applying a convolutional layer.
    
    Parameters
    --------------
    size        : Int, width or height of the input.
    kernel_size : Int, the kernel size of the conv layer (default=5)
    stride      : Int, the stride used in the conv layer (default=2)
    """
    
    return (size - (kernel_size - 1) - 1) // stride  + 1

class DQNetwork(nn.Module):
    
    def __init__(self, w = 120, h = 160, init_zeros = False, stack_size = 4,out = 3):
        """
        Description
        ---------------
        Constructor of Deep Q-network class.
        
        Parameters
        ---------------
        w          : Int, input width (default=120)
        h          : Int, input height (default=160)
        init_zeros : Boolean, whether to initialize the weights to zero or not.
        stack_size : Int, input dimension which is the number of frames to stack to create motion (default=4)
        out        : Int, the number of output units, it corresponds to the number of possible actions (default=3).
                     Be careful, it must be changed when considering a different number of possible actions.
        """
        
        super(DQNetwork, self).__init__()
        
        # Conv Module
        self.conv_1 = nn.Conv2d(in_channels = stack_size, out_channels = 32, kernel_size = 8, stride = 4)
        self.conv_2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 4, stride = 2)
#         self.conv_3 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 5, stride = 2)
        if init_zeros:
            nn.init.constant_(self.conv_1.weight, 0.0)
            nn.init.constant_(self.conv_2.weight, 0.0)
            nn.init.constant_(self.conv_3.weight, 0.0)
        
        convw = conv2d_size_out(conv2d_size_out(w, 8, 4), 4, 2) # width of last conv output
        convh = conv2d_size_out(conv2d_size_out(h, 8, 4), 4, 2) # height of last conv output
        linear_input_size = convw * convh * 64
        
        self.fc = nn.Linear(linear_input_size, 512)
        self.output = nn.Linear(512, out)
        
    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = F.relu(self.conv_2(x))
#         x = F.relu(self.conv_3(x))
        x = F.relu(self.fc(x.view(x.size(0), -1)))
        return self.output(x)
    
class DDDQNetwork(nn.Module):
    
    def __init__(self, w = 120, h = 160, init_zeros = False, stack_size = 4, out = 3):
        """
        Description
        ---------------
        Constructor of Deep Dueling Q-network class.
        
        Parameters
        ---------------
        w          : Int, input width (default=120)
        h          : Int, input height (default=160)
        init_zeros : Boolean, whether to initialize the weights to zero or not.
        stack_size : Int, input dimension which is the number of frames to stack to create motion (default=4)
        out        : Int, the number of output units, it corresponds to the number of possible actions (default=3).
                     Be careful, it must be changed when considering a different number of possible actions.
        """
        
        super(DDDQNetwork, self).__init__()
        
        # Conv Module
        self.conv_1 = nn.Conv2d(in_channels = stack_size, out_channels = 32, kernel_size = 8, stride = 4)
        self.conv_2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 4, stride = 2)
#         self.conv_3 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 5, stride = 2)
        
        convw = conv2d_size_out(conv2d_size_out(w, 8, 4), 4, 2) # width of last conv output
        convh = conv2d_size_out(conv2d_size_out(h, 8, 4), 4, 2) # height of last conv output
        linear_input_size = convw * convh * 64
        
        # Value Module
        self.value_fc_1 = nn.Linear(linear_input_size, 512)
#         self.value_fc_2 = nn.Linear(1024, 512)
        self.value_output = nn.Linear(512, 1)
        
        # Advantage Module
        self.advantage_fc_1 = nn.Linear(linear_input_size, 512)
#         self.advantage_fc_2 = nn.Linear(1024, 512)
        self.advantage_output = nn.Linear(512, out)
        
        
        if init_zeros:
            # Initialize conv module layers to 0
            nn.init.constant_(self.conv_1.weight, 0.0)
            nn.init.constant_(self.conv_2.weight, 0.0)
            nn.init.constant_(self.conv_3.weight, 0.0)
            
            # Initialize value module layers to 0
            nn.init.constant_(self.value_fc_1.weight, 0.0)
            nn.init.constant_(self.value_fc_2.weight, 0.0)
            nn.init.constant_(self.value_output.weight, 0.0)
            
            # Initialize advantage module layers to 0
            nn.init.constant_(self.advantage_fc_1.weight, 0.0)
            nn.init.constant_(self.advantage_fc_2.weight, 0.0)
            nn.init.constant_(self.advantage_output.weight, 0.0)
        
    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = F.relu(self.conv_2(x))
#         x = F.relu(self.conv_3(x))
        x_flat = x.view(x.size(0), -1)
        
        # Through Value Module
        x_value = F.relu(self.value_fc_1(x_flat))
#         x_value = F.relu(self.value_fc_2(x_value))
        x_value = self.value_output(x_value)
        
        # Through Advantage Module
        x_advantage = F.relu(self.advantage_fc_1(x_flat))
#         x_advantage = F.relu(self.advantage_fc_2(x_advantage))
        x_advantage = self.advantage_output(x_advantage)
        
        return x_value + (x_advantage - torch.mean(x_advantage, 1, True))

        
    

























