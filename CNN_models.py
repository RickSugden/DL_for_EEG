 
import torch
import torch.nn as nn
from torch.autograd import Variable 
import torch.nn.functional as F
import math

class PD_CNN(nn.Module):

    def __init__(self,chunk_size=2500):
        super(PD_CNN, self).__init__()
        self.chunk_size = chunk_size

        self.conv1 = nn.Conv1d(in_channels=60, out_channels=21, kernel_size=20,stride=1)
        self.norm1 = nn.BatchNorm1d(num_features=21)
        self.maxpool1 = nn.MaxPool1d(kernel_size=4,stride=4)

        self.conv2 = nn.Conv1d(in_channels=21, out_channels=42, kernel_size=10,stride=1)
        self.norm2 = nn.BatchNorm1d(num_features=42)
        self.maxpool2 = nn.MaxPool1d(kernel_size=4,stride=4)

        self.conv3 = nn.Conv1d(in_channels=42, out_channels=42, kernel_size=10,stride=1)
        self.norm3 = nn.BatchNorm1d(num_features=42)
        self.maxpool3 = nn.MaxPool1d(kernel_size=4,stride=4)

        self.conv4 = nn.Conv1d(in_channels=42, out_channels=64, kernel_size=5,stride=1)
        self.norm4 = nn.BatchNorm1d(num_features=64)
        self.maxpool4 = nn.MaxPool1d(kernel_size=4,stride=4)

        
        self.relu = nn.LeakyReLU(0.1)

        
        self.fc1 = nn.Linear(in_features=448,out_features=256)#in_features=4*(self.chunk_size-8)
        self.dropout1 = nn.Dropout(p=0.5)

        self.fc2 = nn.Linear(in_features=256, out_features=64)
        self.dropout2 = nn.Dropout(p=0.5)

        self.fc3 = nn.Linear(in_features=64, out_features=16)
        self.fc4 = nn.Linear(in_features=16, out_features=2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        x = self.relu(self.maxpool1(self.norm1(self.conv1(x))))

        x = self.relu(self.maxpool2(self.norm2(self.conv2(x))))
        
        x = self.relu(self.maxpool3(self.norm3(self.conv3(x))))
        
        x = self.relu(self.maxpool4(self.norm4(self.conv4(x))))
        
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        
        x = self.dropout1(self.fc1(x))
        x = self.dropout2(self.fc2(x))
        x = self.fc3(x)

        x = self.softmax(self.fc4(x))
        return x


class PD_LSTM(nn.Module):
  def __init__(self, input_size=60, hidden_size=6, num_layers=1, seq_length=2500, num_classes=2, device='cpu'):
    super(PD_LSTM, self).__init__()
    self.num_classes = num_classes #number of classes
    self.num_layers = num_layers #number of layers
    self.input_size = input_size #input size
    self.hidden_size = hidden_size #hidden state
    self.seq_length = seq_length #sequence length
    self.device = device

    self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True) #lstm
    self.dropout = nn.Dropout(p=0.2, inplace=False)
    #self.lstm = nn.LSTM(input_size=64, hidden_size=64, num_layers=1, batch_first=True) #lstm
    self.fc_1 =  nn.Linear(hidden_size, 32) #fully connected 1
    self.fc = nn.Linear(32, num_classes) #fully connected last layer

    self.relu = nn.ReLU()
    self.softmax = nn.Softmax(dim=1)

  def forward(self, x):
      
    #initialize the hidden and internal state to be all zeros
    h_0 = Variable(torch.rand(self.num_layers, x.size(0), self.hidden_size)).to(self.device) #hidden state
    c_0 = Variable(torch.rand(self.num_layers, x.size(0), self.hidden_size)).to(self.device) #internal state

    #permute data to (batch_first, seq, feature)
    x = x.permute(0, 2, 1)

    # Propagate input through LSTM
    output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state


    hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
    out = self.relu(hn)

    out = self.dropout(out)

    out = self.fc_1(out) #first Dense
    out = self.relu(out) #relu
    
    out = self.fc(out) #Final Output
    output = self.softmax(out)
    return output
  

class EEGNet(nn.Module):
    def __init__(self, num_channels, time_points):
        super(EEGNet, self).__init__()
        self.T = time_points

        # Layer 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1, 60), padding=0)
        self.batchnorm1 = nn.BatchNorm2d(16, False)

        # Layer 2
        self.padding1 = nn.ZeroPad2d((16, 17, 0, 1))
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(2, 30))
        self.batchnorm2 = nn.BatchNorm2d(4, False)
        self.pooling2 = nn.MaxPool2d((2, 4))

        # Layer 3
        self.padding2 = nn.ZeroPad2d((2, 1, 4, 3))
        self.conv3 = nn.Conv2d(4, 4, (8, 2))
        self.batchnorm3 = nn.BatchNorm2d(4, False)
        self.pooling3 = nn.MaxPool2d((2, 4))

        # FC Layer
        #self.fc1 = None  # We'll initialize this later
        self.fc1 = nn.Linear(2512, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 2)
        self.softmax= nn.Softmax(dim=1)


    def forward(self, x):
        
        x = x.view(x.shape[0], 1, x.shape[2],  x.shape[1])  # Reshape the input       
        # Layer 1
        x = F.elu(self.conv1(x))
        x = self.batchnorm1(x)
        x = F.dropout(x, 0.25)
        x = x.permute(0, 3, 1, 2)  # Revised permute operation
    
      
        # Layer 2       
        x = self.padding1(x)
        x = F.elu(self.conv2(x))
        x = self.batchnorm2(x)       
        x = F.dropout(x, 0.25)
        x = self.pooling2(x)
   
        # Layer 3
        x = self.padding2(x) 
        x = F.elu(self.conv3(x))
        x = self.batchnorm3(x)       
        x = F.dropout(x, 0.25)
        x = self.pooling3(x)      
        # Flatten the output from the last layer  
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
      
        # FC Layer
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.softmax(self.fc3(x))

        return x

class ResNet(nn.Module):
    def __init__(self,n_in=60,n_classes=2,time_steps=2500):
        super(ResNet,self).__init__()
        self.n_in = n_in
        self.n_classes = n_classes
        self.time_steps = time_steps

        blocks  = [n_in,8,16,16]
        self.blocks = nn.ModuleList()
        for b,_ in enumerate(blocks[:-1]):
            self.blocks.append(ResidualBlock(*blocks[b:b+2],self.time_steps))
        
        self.fc1 =  None
        self.fc2 =  nn.Linear(512,64)
        self.fc3 =  nn.Linear(64,2)

        self.softmax= nn.Softmax(dim=1)

      
        
    def forward(self, x: torch.Tensor):

        for block in self.blocks:
            x = block(x)
     

            
        x = x.view(x.size(0), -1)


        if self.fc1 == None: 
          self.fc1 = nn.Linear(x.shape[-1],512).cuda()
        x = self.fc1(x)

        x = self.fc2(x)
     
        x = self.fc3(x)
     
        
        x = self.softmax(x)
        
        return x #.view(-1,self.n_classes)
    
class ResidualBlock(nn.Module):
    def __init__(self,in_maps,out_maps,time_steps):
        super(ResidualBlock,self).__init__()
        self.in_maps  = in_maps
        self.out_maps = out_maps
        self.time_steps = time_steps
        
        self.conv1 = nn.Conv1d(self.in_maps, self.out_maps,7,padding=3, stride=2)
        self.bn1   = nn.BatchNorm1d(self.out_maps)

        self.conv2 = nn.Conv1d(self.out_maps,self.out_maps,5,padding=2)
        self.bn2   = nn.BatchNorm1d(self.out_maps)
        
        self.conv3 = nn.Conv1d(self.out_maps,self.out_maps,3,padding=1)
        self.bn3   = nn.BatchNorm1d(self.out_maps)

        
    def forward(self,x):
        x   = F.relu(self.bn1(self.conv1(x)))
        inx = x
        x   = F.relu(self.bn2(self.conv2(x)))
        x   = F.relu(self.bn3(self.conv3(x))+inx)
        
        return x


class Conv1DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Conv1DBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2))
        self.bn = nn.BatchNorm1d(out_channels)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x)


class VGG13(nn.Module):
    def __init__(self, num_channels, num_filters,  output_nums=2, dropout_rate=False):
        super(VGG13, self).__init__()
        self.output_nums = output_nums
        self.dropout_rate = dropout_rate
        self.num_channels = num_channels
        self.num_filters = num_filters

        self.block1 = nn.Sequential(
            Conv1DBlock(num_channels, num_filters * (2 ** 0), 3),
            Conv1DBlock(num_filters * (2 ** 0), num_filters * (2 ** 0), 3),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        self.block2 = nn.Sequential(
            Conv1DBlock(num_filters * (2 ** 0), num_filters * (2 ** 1), 3),
            Conv1DBlock(num_filters * (2 ** 1), num_filters * (2 ** 1), 3),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        self.block3 = nn.Sequential(
            Conv1DBlock(num_filters * (2 ** 1), num_filters * (2 ** 2), 3),
            Conv1DBlock(num_filters * (2 ** 2), num_filters * (2 ** 2), 3),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        self.block4 = nn.Sequential(
            Conv1DBlock(num_filters * (2 ** 2), num_filters * (2 ** 3), 3),
            Conv1DBlock(num_filters * (2 ** 3), num_filters * (2 ** 3), 3),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        self.block5 = nn.Sequential(
            Conv1DBlock(num_filters * (2 ** 3), num_filters * (2 ** 3), 3),
            Conv1DBlock(num_filters * (2 ** 3), num_filters * (2 ** 3), 3),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # We will initialize fc later, once we know the input dimension
        self.fc = None  

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        if self.fc is None:
            # Initialize fc during the first forward pass
            n_features = x.shape[1] * x.shape[2]
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(n_features, 4096),
                nn.ReLU(),
                nn.Linear(4096, 4096),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate) if self.dropout_rate else nn.Identity(),
                nn.Linear(4096, self.output_nums)
            ).to(x.device)

        x = self.fc(x)
        
        x = F.softmax(x, dim=1)
        return x
    

class DeepConvNet(torch.nn.Module):
    def __init__(self, n_output=2):
        super(DeepConvNet, self).__init__()
        self.filters1 = 1
        self.filters2 = 2
        self.filters3 = 4
        self.filters4 = 8
        self.block1 = nn.Sequential(
            # Conv2d(1, 25, kernel_size=(1,5),padding='VALID',bias=False),
            # Conv2d(25, 25, kernel_size=(2,1), padding='VALID',bias=False),
            nn.Conv2d(1, self.filters1, kernel_size=(1,5),bias=False),
            nn.Conv2d(self.filters1, self.filters1, kernel_size=(2,1),bias=False),
            nn.BatchNorm2d(self.filters1, eps=1e-05, momentum=0.1),
            nn.LeakyReLU(negative_slope=0.04),
            nn.MaxPool2d(kernel_size=(1,4)),
            nn.Dropout(p=0.47)
        )

        self.block2 = nn.Sequential(
            # Conv2d(25, 50, kernel_size=(1,5),padding='VALID',bias=False),
            nn.Conv2d(self.filters1, self.filters2, kernel_size=(1,5),bias=False),
            nn.BatchNorm2d(self.filters2, eps=1e-05, momentum=0.1),
            nn.LeakyReLU(negative_slope=0.09),
            nn.MaxPool2d(kernel_size=(1,4)),
            nn.Dropout(p=0.47),
        )
        self.block3 = nn.Sequential(
            # Conv2d(50, 100, kernel_size=(1,5),padding='VALID',bias=False),
            nn.Conv2d(self.filters2, self.filters3, kernel_size=(1,5),bias=False),
            nn.BatchNorm2d(self.filters3, eps=1e-05, momentum=0.1),
            nn.LeakyReLU(negative_slope=0.04),
            nn.MaxPool2d(kernel_size=(1,4)),
            nn.Dropout(p=0.47),
        )
        self.block4 = nn.Sequential(
            # Conv2d(100, 200, kernel_size=(1,5),padding='VALID',bias=False),
            nn.Conv2d(self.filters3, self.filters4 , kernel_size=(1,5),bias=False),
            nn.BatchNorm2d(self.filters4 , eps=1e-05, momentum=0.1),
            nn.LeakyReLU(negative_slope=0.09),
            nn.MaxPool2d(kernel_size=(1,4)),
            nn.Dropout(p=0.47),
        )
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3776,512, bias=True),
            nn.Linear(512,64, bias=True),
            nn.Linear(64,2, bias=True),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = x.view(x.shape[0], 1, x.shape[1], x.shape[2])
        x = self.block1(x)
        #print(x.shape)
        x = self.block2(x)
        #print(x.shape)
        x = self.block3(x)
        #print(x.shape)
        x = self.block4(x)
        #print(x.shape)
        x = self.mlp(x)
        #print(x.shape)
        return x
    

