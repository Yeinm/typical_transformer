import copy

import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torchtext



class Embedding(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embedding, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

d_model = 512
vocab = 1000

x = Variable(torch.LongTensor([[100, 2, 421, 508],[491, 998, 1, 221]]))

emb= Embedding(d_model, vocab)

embr =emb(x)
x=embr

'''
print('embr:',embr)
print('embr_shape:',embr.shape)
'''

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        #这里变成max_len,1的矩阵
        position = torch.arange(0, max_len).unsqueeze(1)


        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                          requires_grad=False)
        return self.dropout(x)


pe = PositionalEncoding(d_model,dropout=0.1,max_len=60)
pe_result=pe(x)
#print(pe_result.size(),'context',pe_result)

'''
plt.figure(figsize=(15, 5))
pe = PositionalEncoding(20,0)
y = pe(Variable(torch.zeros(1, 100, 20)))
plt.plot(np.arange(0, 100), y[0, :, 4:8].data.numpy())
plt.legend(['dim %d'%p for p in [4,5,6,7]])
plt.show()
'''

def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape),k= 1).astype('uint8')
    return torch.from_numpy(1-subsequent_mask)

def attention(query, key, value, mask=None, dropout= None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim= -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn
'''
input = Variable(torch.randn(5,5))
mask = Variable(torch.zeros(5,5))
input.masked_fill_(mask == 0, -1e9)
'''

query = key = value = pe_result
'''
attn, p_attn = attention(query, key, value)

print(attn)
print(p_attn)
'''
'''
mask = Variable(torch.zeros(2,4,4))
attn, p_attn = attention(query, key, value,mask=mask)
print(attn)
print(p_attn)
'''

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        #测试中常用的assert
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

#OK let we test it

head = 8
embedding_dim = 512
dropout = 0.1
#query = key = value = pe_result

mask = Variable(torch.zeros(2,4,4))
mha = MultiHeadedAttention(head,embedding_dim,dropout)
mha_result = mha(query,key,value,mask)
'''
print(mha_result)
print(mha_result.shape)
'''

#前馈全连接层
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

d_model = 512
d_ff = 2048
dropout = 0.1
x = mha_result
ff = PositionwiseFeedForward(d_model,d_ff,dropout)
ff_result = ff(x)

#(ff_result)

#规范化层
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

features = d_model = 512
x = ff_result
ln = LayerNorm(features)
ln_result = ln(x)
#print(ln_result)



#残差连接实现部分
class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
'''

dropout = 0.2
head = 8
d_model = 512
x = pe_result
mask = Variable(torch.zeros(2,4,4))
self_attn = MultiHeadedAttention(head,d_model,dropout)
sublayer = lambda x: self_attn(x,x,x,mask)

size = 512
sc = SublayerConnection(size,dropout)

sc_result = sc(x,sublayer)
print(sc_result)
print(sc_result.shape)
'''

#现在是总的编码器层
class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)



'''
size = 512
head = 8
d_model = 512
d_ff = 64
x = pe_result
dropout= 0.2
self_attn = MultiHeadedAttention(head,d_model)
ff = PositionwiseFeedForward(d_model,d_ff,dropout)
mask = Variable(torch.zeros(2,4,4))
el = EncoderLayer(size,self_attn,ff,dropout)
el_result = el(x,mask)
print(el_result)
print(el_result.shape)
'''

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

size = 512
head = 8
d_model = 512
d_ff = 64
c = copy.deepcopy
attn = MultiHeadedAttention(head,d_model)
ff = PositionwiseFeedForward(d_model,d_ff,dropout)
dropout = 0.2
layer = EncoderLayer(size,c(attn),c(ff),dropout)
N = 8
mask = Variable(torch.ones(2,4,4))

en = Encoder(layer,N)
en_result = en(x,mask)
'''
print(en_result)
print(en_result.shape)
'''

class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

size = 512
head = 8
d_model = 512
d_ff = 64
dropout = 0.2
self_attn = src_attn = MultiHeadedAttention(head,d_model,dropout)
ff = PositionwiseFeedForward(d_model,d_ff,dropout)

x = pe_result
memory = en_result


mask = Variable(torch.ones(2,4,4))
source_mask = target_mask = mask

dl = DecoderLayer(size,self_attn,src_attn,ff,dropout)
dl_result = dl(x,memory,source_mask,target_mask)
'''
print(dl_result)
print(dl_result.shape)
'''


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

layer = DecoderLayer(d_model,c(attn),c(attn),c(ff),dropout)
N=8

mask = Variable(torch.zeros(2,4,4))
source_mask = target_mask = mask
de = Decoder(layer,N)
de_result = de(x,memory,source_mask,target_mask)
'''
print(de_result)
print(de_result.shape)
'''

class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

gen = Generator(d_model,vocab)
gen_result = gen(de_result)
'''
print(gen_result)
print(gen_result.shape)
'''

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


vocab_size = 1000
d_model = 512
encoder = en
decoder = de
source_embed = nn.Embedding(vocab_size, d_model)
tgt_embed = nn.Embedding(vocab_size, d_model)
generator = gen

source = target = Variable(torch.LongTensor([[100,2,421,508],[491,998,1,221]]))
source_mask = target_mask = Variable(torch.zeros(2,4,4))

ed = EncoderDecoder(encoder, decoder, source_embed, tgt_embed, generator)
ed_result = ed(source,target,source_mask,target_mask)
'''
print(ed_result)
print(ed_result.shape)
'''

def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters~"
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embedding(d_model, src_vocab), c(position)),
        nn.Sequential(Embedding(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab)
    )
    for param in model.parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)
    return model
'''
source_vocab_size = 11
target_vocab_size = 11
N=6
model = make_model(source_vocab_size, target_vocab_size,N)

print(model)
'''
















































