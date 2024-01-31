
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
    


class PostionalEncoding(nn.Module): 
    def __init__(
        self, 
        dropout: float=0.1, 
        max_seq_len: int=5000, 
        d_model: int=512,
        batch_first: bool=False    ): 
        super().__init__()

        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout) 
        self.batch_first = batch_first 
        self.x_dim = 1 if batch_first else 0  
        position = torch.arange(max_seq_len).unsqueeze(1) 
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)) 
        pe = torch.zeros(max_seq_len, 1, d_model)
        
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x): 
        x = x + self.pe[:x.size(self.x_dim)]

        x = self.dropout(x)  
        return x

class ClassificationHead(nn.Module):
    def __init__(self,d_model, seq_len , details, n_classes: int = 5):
      super().__init__()
      self.norm = nn.LayerNorm(d_model)
      self.details = details
      #self.flatten = nn.Flatten()
      self.seq = nn.Sequential( nn.Flatten() , nn.Linear(d_model*seq_len , 512) ,nn.ReLU(),nn.Linear(512, 256) # nn.Linear(d_model * seq_len , 512)
                               ,nn.ReLU(),nn.Linear(256, 128),nn.ReLU(),nn.Linear(128, n_classes))
 
    def forward(self,x):

      if self.details:  print('in classification head : '+ str(x.size())) 
      x= self.norm(x)
      #x= self.flatten(x)
      x= self.seq(x)
      if self.details: print('in classification head after seq: '+ str(x.size())) 
      return x


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        # '-1' means last dimension. 

        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_head, details):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention( details=details)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)
        self.details = details

    def forward(self, q, k, v ):
        # 1. dot product with weight matrices

        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        if self.details: print('in Multi Head Attention Q,K,V: '+ str(q.size()))
        # 2. split tensor by number of heads
        q, k, v = self.split(q), self.split(k), self.split(v)

        if self.details: print('in splitted Multi Head Attention Q,K,V: '+ str(q.size()))
        # 3. do scale dot product to compute similarity
        out, attention = self.attention(q, k, v )
        
        if self.details: print('in Multi Head Attention, score value size: '+ str(out.size()))
        # 4. concat and pass to linear layer
        out = self.concat(out)
        out = self.w_concat(out)

        # 5. visualize attention map
        # TODO : we should implement visualization

        if self.details: print('in Multi Head Attention, score value size after concat : '+ str(out.size()))
        return out

    def split(self, tensor):
        """
        split tensor by number of head

        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.size() 
        d_tensor = d_model // self.n_head #this makes sure that d_k, d_model and d_head all fit together. 
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        # it is similar with group convolution (split by number of heads)

        return tensor

    def concat(self, tensor):
        """
        inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class ScaleDotProductAttention(nn.Module):
    """
    compute scale dot product attention

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeury(encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self, details):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.details = details
    def forward(self, q, k, v ,e=1e-12):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size()
        
        # 1. dot product Query with Key^T to compute similarity
        k_t = k.transpose(2, 3)  # transpose
        
        if self.details: print('in Scale Dot Product, k_t size: '+ str(k_t.size()))
        score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product


        if self.details: print('in Scale Dot Product, score size: '+ str(score.size()))
        # 3. pass them softmax to make [0, 1] range
        score = self.softmax(score)

        if self.details: print('in Scale Dot Product, score size after softmax : '+ str(score.size()))

        if self.details: print('in Scale Dot Product, v size: '+ str(v.size()))
        # 4. multiply with Value
        v = score @ v

        if self.details: print('in Scale Dot Product, v size after matmul: '+ str(v.size()))
        return v, score

class EncoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob,details):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head, details=details)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.details = details
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x):
        # 1. compute self attention
        _x = x
        x = self.attention(q=x, k=x, v=x )
        
        if self.details: print('in encoder layer : '+ str(x.size()))
        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)
        
        if self.details: print('in encoder after norm layer : '+ str(x.size()))
        # 3. positionwise feed forward network
        _x = x
        x = self.ffn(x)
      
        if self.details: print('in encoder after ffn : '+ str(x.size()))
        # 4. add and norm
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x
    
class Encoder(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, n_layers, drop_prob,details):
        super().__init__()

        
        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head
                                                  ,details=details,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

    def forward(self, x ): 
        for layer in self.layers:
            x = layer(x ) 
        return x
    
class Transformer(nn.Module):

    def __init__(self,device, d_model=60, n_head=4, max_len=5000, seq_len=200,
                 ffn_hidden=128, n_layers=2, drop_prob=0.1, details =False):
        super().__init__() 
        print('in transformer, d_model: '+ str(d_model), 'n_head: '+ str(n_head), 'max_len: '+ str(max_len), 'seq_len: '+ str(seq_len),
                 'ffn_hidden: '+ str(ffn_hidden), 'n_layers: '+ str(n_layers), 'drop_prob: '+ str(drop_prob), 'details: '+ str(details))
        #self.device = device
        self.seq_len = seq_len
        self.details = details 
        self.encoder_input_layer = nn.Linear(   
            in_features=60, 
            out_features=d_model 
            )
   
        self.pos_emb = PostionalEncoding( max_seq_len=max_len,batch_first=False, d_model=d_model, dropout=0.1) 
        self.encoder = Encoder(d_model=d_model,
                               n_head=n_head, 
                               ffn_hidden=ffn_hidden, 
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               details=details,
                               )
        self.classHead = ClassificationHead(seq_len=seq_len,d_model=d_model,details=details,n_classes=2)
   

    def forward(self, src ): 
        src = src.permute(0,2,1)
        if src.size(1) > self.seq_len:
            #select only the first seq_len elements
            src = src[:,:self.seq_len,:]

        if self.details: print('before input layer: '+ str(src.size()) )
        src= self.encoder_input_layer(src)
        if self.details: print('after input layer: '+ str(src.size()) )
        src= self.pos_emb(src)
        if self.details: print('after pos_emb: '+ str(src.size()) )
        enc_src = self.encoder(src) 
        cls_res = self.classHead(enc_src)
        if self.details: print('after cls_res: '+ str(cls_res.size()) )
        return cls_res