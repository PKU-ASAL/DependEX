import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

class EncoderDecoder(nn.Module):
	def __init__(self, encoder, src_embed, generator):
		super(EncoderDecoder, self).__init__()
		self.encoder = encoder
		self.src_embed = src_embed
		self.generator = generator
		
	def forward(self, src, src_mask):
		out = self.encode(src, src_mask)
		return out
	
	def encode(self, src, src_mask):
		return self.encoder(self.src_embed(src), src_mask)

class Generator(nn.Module):
	def __init__(self, d_model, class_num):
		super(Generator, self).__init__()
		self.class_num = class_num
		self.proj = nn.Linear(d_model, 40)
		self.batchnorm1 = nn.BatchNorm1d(49)
		self.preluip1 = nn.ReLU()
		self.batchnorm2 = nn.BatchNorm1d(30)
		self.ip1 = nn.Linear(49 * 40, 30)
		self.ip2 = nn.Linear(30,self.class_num)
		self.dropout = nn.Dropout(0.1)

	def forward(self, x):
		proj_x = self.proj(x)
		proj_x = proj_x.view(-1, x.shape[1] * 40)
		ip1 = self.ip1(proj_x)
		ip1_relu = self.preluip1(ip1)
		ip2 = self.ip2(ip1_relu)
		return ip1,F.log_softmax(ip2, dim=-1)

def clones(module, N):
	return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

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

class SublayerConnection(nn.Module):
	def __init__(self, size, dropout):
		super(SublayerConnection, self).__init__()
		self.norm = LayerNorm(size)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x, sublayer):
		return x + self.dropout(sublayer(self.norm(x)))

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

def attention(query, key, value, mask=None, dropout=None):
	d_k = query.size(-1)	
	scores = torch.matmul(query, key.transpose(-2, -1)) \
			 / math.sqrt(d_k)
	if mask is not None:
		scores = scores.masked_fill(mask == 0, -1e9)
	p_attn = F.softmax(scores, dim = -1)
	if dropout is not None:
		p_attn = dropout(p_attn)
	return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
	def __init__(self, h, d_model, dropout=0.1):
		super(MultiHeadedAttention, self).__init__()
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
		query, key, value = \
			[l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
			 for l, x in zip(self.linears, (query, key, value))]
		x, self.attn = attention(query, key, value, mask=mask,dropout=self.dropout)
		x = x.transpose(1, 2).contiguous() \
			 .view(nbatches, -1, self.h * self.d_k)
		return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
	def __init__(self, d_model, d_ff, dropout=0.1):
		super(PositionwiseFeedForward, self).__init__()
		self.w_1 = nn.Linear(d_model, d_ff)
		self.w_2 = nn.Linear(d_ff, d_model)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		tmp = self.w_2(self.dropout(F.relu(self.w_1(x))))
		return tmp

class Embeddings(nn.Module):
	def __init__(self, d_model, vocab):
		super(Embeddings, self).__init__()
		self.lut = nn.Embedding(vocab, d_model)
		self.d_model = d_model

	def forward(self, x):
		return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
	def __init__(self, d_model, dropout, max_len=5000):
		super(PositionalEncoding, self).__init__()
		self.dropout = nn.Dropout(p=dropout)
		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0.0, max_len).unsqueeze(1)
		div_term = torch.exp(torch.arange(0.0, d_model, 2) *
							 -(math.log(10000.0) / d_model))
		pe[:, 0::2] = torch.sin(position * div_term).type(torch.LongTensor)
		pe[:, 1::2] = torch.cos(position * div_term).type(torch.LongTensor)
		pe = pe.unsqueeze(0)
		self.register_buffer('pe', pe)
		
	def forward(self, x):
		x = x + Variable(self.pe[:, :x.size(1)],requires_grad=False)
		return self.dropout(x)

class Transformer(nn.Module):
	def make_model(self, class_num, N=6,d_model=512, d_ff=2048, h=8, dropout=0.1):
		c = copy.deepcopy
		attn = MultiHeadedAttention(h, d_model)
		ff = PositionwiseFeedForward(d_model, d_ff, dropout)
		position = PositionalEncoding(d_model, dropout)
		model = EncoderDecoder(
		Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
		nn.Sequential(c(position)),
		Generator(d_model, class_num))
		for p in model.parameters():
			if p.dim() > 1:
				nn.init.xavier_uniform_(p)
		return model

	def __init__(self, args):
		super(Transformer, self).__init__()
		self.att_size = args.img_fatures_size
		self.embed_size = args.embed_size
		self.use_bn = args.use_bn
		self.drop_prob_lm = args.drop_prob_lm
		self.class_num = args.class_num
		self.num_layers = args.num_layers
		self.ff_size = args.ff_size
		self.att_embed = nn.Sequential(*(((nn.BatchNorm1d(self.att_size),) if self.use_bn else ())+(nn.Linear(self.att_size, self.embed_size), nn.ReLU(),nn.Dropout(self.drop_prob_lm))+((nn.BatchNorm1d(self.embed_size),) if self.use_bn==2 else ())))
		self.model = self.make_model(self.class_num, self.num_layers, self.embed_size,self.ff_size, 8, self.drop_prob_lm)

	def forward(self, att_feats, att_masks = None):
		att_feats = self.att_embed(att_feats)
		out = self.model(att_feats, att_masks)
		outputs = self.model.generator(out)
		return outputs
