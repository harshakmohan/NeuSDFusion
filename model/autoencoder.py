import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, num_layers, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, num_heads, dim_feedforward=d_model * 4, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.d_model = d_model

    def forward(self, src):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)
        return memory

class TransformerDecoder(nn.Module):
    def __init__(self, d_model, output_dim, num_heads, num_layers, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Linear(d_model, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        decoder_layers = nn.TransformerDecoderLayer(d_model, num_heads, dim_feedforward=d_model * 4, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers)
        self.output_layer = nn.Linear(d_model, output_dim)
        self.d_model = d_model

    def forward(self, tgt, memory):
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt)
        output = self.transformer_decoder(tgt, memory)
        output = self.output_layer(output)
        return output

class TransformerAutoEncoder(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, num_layers, dropout=0.1):
        super(TransformerAutoEncoder, self).__init__()
        self.encoder = TransformerEncoder(input_dim, d_model, num_heads, num_layers, dropout)
        self.decoder = TransformerDecoder(d_model, input_dim, num_heads, num_layers, dropout)

    def forward(self, src):
        memory = self.encoder(src)
        output = self.decoder(src, memory)
        return output

# Example usage
input_dim = 512
d_model = 512
num_heads = 8
num_layers = 6
dropout = 0.1

autoencoder = TransformerAutoEncoder(input_dim, d_model, num_heads, num_layers, dropout)

# Example input
src = torch.randn(10, 32, input_dim)  # (sequence_length, batch_size, input_dim)
output = autoencoder(src)
print(output.shape)  # Should match input shape: (sequence_length, batch_size, input_dim)
