 
import torch
import torch.nn as nn
from torch.autograd import Variable 
import torch.nn.functional as F
import math
from collections import OrderedDict


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



'''
Code to test model of Attention-based Transformer model.
'''
class FeedForwardLayer(nn.Module):
    # Class encapsulating the Feed forward layer within the attention block. 
    def __init__(self, input_size, hidden_size, output_size):
        super(FeedForwardLayer, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.GELU = nn.GELU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.GELU(self.fc1(x))
        x = self.fc2(x)
        return x


class AttentionBlock(nn.Module):
    def __init__(self, model_dim=60, heads=4): # 60 and 4 are default values to build the model with.
        super(AttentionBlock, self).__init__()

        self.MHA = nn.MultiheadAttention(embed_dim=model_dim, num_heads=heads, batch_first=True)
        self.norm1 = nn.LayerNorm(model_dim)

        self.feedforward = FeedForwardLayer(model_dim, model_dim, model_dim)
        self.norm2 = nn.LayerNorm(model_dim)


    def forward(self, x):
        x0 = self.MHA(x, x, x, need_weights=False)
        x = self.norm1(x + x0[0])

        x1 = self.feedforward(x)
        x = self.norm2(x+x1)

        return x
    
#model 2 - Attention-based Transformer
class transformNET(nn.Module):
    def __init__(self, num_blocks = 6, heads=4, model_dim=60):
        super(transformNET, self).__init__()

        self.blocks = OrderedDict()
        for i in range((num_blocks)):
          self.blocks[str(i)] = AttentionBlock(model_dim, heads)

        self.transformer_encoder = nn.Sequential(self.blocks)


        self.maxpool1 = nn.MaxPool1d(kernel_size=8, stride=8)

        self.gelu = nn.GELU()
        # Adjusted dimensions according to your model's requirement
        self.fc1 = nn.Linear(3840, 256)  # Adjust the input size as needed
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, 64)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(64, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        x = self.transformer_encoder(x)

        x = torch.flatten(x, 1)
        x = self.maxpool1(x)

        x = self.gelu(self.fc1(x))
        x = self.dropout1(x)
        x = self.gelu(self.fc2(x))
        x = self.dropout2(x)
        x = self.gelu(self.fc3(x))
        x = self.softmax(x)
        return x