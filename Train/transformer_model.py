import torch.nn as nn
import torch
import torch.nn.functional as F
import math
# This code is partially written based on "The Annotated Transformer" 
# as well as code I have written for my Vision Transformer implementation


# ----- Attention ----- #
class SelfAttention(nn.Module):
    def __init__(self, embed_dim, key_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.key_dim = key_dim
        
        # Use one matrix for Key, Query, Value weight matrices
        self.Wq = nn.Parameter(torch.empty(embed_dim, key_dim))
        self.Wk = nn.Parameter(torch.empty(embed_dim, key_dim))
        self.Wv = nn.Parameter(torch.empty(embed_dim, key_dim))
        self.softmax = nn.Softmax(dim = -1)
    
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.W)
        
    def forward(self, q, k, v, mask=None):
        Q = torch.matmul(q, self.Wq)
        K = torch.matmul(k, self.Wk)
        V = torch.matmul(v, self.Wv)
        
        # Scaled dot product
        dot_product = torch.matmul(Q, K.transpose(-2, -1)) 
        
        # Scaled dot product
        Z = dot_product / (self.key_dim ** 0.5)
        
        # Zero out the padding tokens
        if mask is not None:
            Z = Z.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to obtain attention scores
        score = self.softmax(Z)
        
        # Get weighted values
        output = torch.matmul(score, V)
        
        return output
             
# Multi head attention
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        
        assert embed_dim % num_heads == 0, "Embedding dimension should be divisible with the number of heads"
        self.key_dim = embed_dim // num_heads
        
        # Init multihead-attention
        self.multi_head_attention = nn.ModuleList([SelfAttention(embed_dim, self.key_dim) for _ in range(num_heads)])
        
        # Multihead-attention weight
        self.W = nn.Parameter(torch.empty((embed_dim, embed_dim)))
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.W)
        
    def forward(self, q, k, v,  mask):
        # Compute self-attention scores of each head 
        attention_outputs = [attention(q, k, v, mask) for attention in self.multi_head_attention]
        
        # Concatenate the outputs of all heads
        concat_output = torch.cat(attention_outputs, dim=-1)  # Shape: (batch_size, seq_len, embed_dim)
        
        # Compute multi-head attention score
        attention_output = torch.matmul(concat_output, self.W)  # Shape: (batch_size, seq_len, embed_dim)
        
        return attention_output
    
# ----- Residual and fully-connected layers ----- # 
class ResidualConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    """
    def __init__(self, embed_dim, dropout):
        super(ResidualConnection, self).__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x, sublayer):
        # Apply the sublayer to the normalized input, add the result to the original input
        return x + self.dropout(sublayer(self.norm(x)))
        
class PositionwiseFeedForward(nn.Module):
    """
    Implements a Position-wise Feed-Forward Network (FFN) as described in the Transformer paper.
    """
    def __init__(self, embed_dim, hidden_dim, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(embed_dim, hidden_dim)
        self.w_2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))
        

# ----- Encoder -----#
class EncoderLayer(nn.Module):
    """
    A single layer of the Transformer encoder.
    """
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout=0.1):
        super(EncoderLayer, self).__init__()
        
        # Attention layer
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        
        # Position-wise feed-forward network
        self.feed_forward = PositionwiseFeedForward(embed_dim, hidden_dim, dropout)
        
        # Residual connections for attention and feed-forward layers
        self.skip_connections = nn.ModuleList([
            ResidualConnection(embed_dim, dropout),  # For attention
            ResidualConnection(embed_dim, dropout)   # For feed-forward
        ])
        
    def forward(self, x, mask=None):
        # Apply the attention layer with residual connection
        x = self.skip_connections[0](x, lambda x: self.attention(x, x, x, mask))
        
        # Apply the feed-forward layer with residual connection
        x = self.skip_connections[1](x, self.feed_forward)
        
        return x

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout, N):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(embed_dim, num_heads, hidden_dim, dropout) for _ in range(N)])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
            
        return self.norm(x)
    
# ----- Decoder ----- #
class DecoderLayer(nn.Module):
    """
    A single layer of the Transformer decoder.
    """
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout=0.1):
        super(DecoderLayer, self).__init__()
        
        # Self-attention layer
        self.self_attention = MultiHeadAttention(embed_dim, num_heads)
        
        # Cross-attention layer (attends to encoder output)
        self.source_attention = MultiHeadAttention(embed_dim, num_heads)
        
        # Feed-forward network
        self.feed_forward = PositionwiseFeedForward(embed_dim, hidden_dim, dropout)
        
        # Residual connections for self-attention, cross-attention, and feed-forward
        self.skip_connections = nn.ModuleList([
            ResidualConnection(embed_dim, dropout),  # For self-attention
            ResidualConnection(embed_dim, dropout),  # For cross-attention
            ResidualConnection(embed_dim, dropout),  # For feed-forward
        ])
        
    def forward(self, x, memory, src_mask=None, tgt_mask=None):
        # Self-attention with residual connection
        x = self.skip_connections[0](x, lambda x: self.self_attention(x, x, x, tgt_mask))
        
        # Cross-attention with residual connection
        x = self.skip_connections[1](x, lambda x: self.source_attention(x, memory, memory, src_mask))
        
        # Feed-forward network with residual connection
        x = self.skip_connections[2](x, self.feed_forward)
        
        return x
    
class Decoder(nn.Module):
    """
    The Transformer decoder composed of multiple decoder layers.
    """
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout, N):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(embed_dim, num_heads, hidden_dim, dropout) for _ in range(N)])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, memory, src_mask=None, tgt_mask=None):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


# ----- Embedding ----- #
class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, embed_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * -(math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return x

class Embeddings(nn.Module):
    def __init__(self, vocab, embed_dim, dropout, PAD):
        super(Embeddings, self).__init__()
        self.embed = nn.Embedding(vocab, embed_dim, padding_idx=PAD)
        
        self.dropout = nn.Dropout(p=dropout)
        
        self.positional_encodings = PositionalEncoding(embed_dim)

    def forward(self, x):
        # Embed each token
        x = self.embed(x)

        # Get the positional embeddings for the input sequence
        x = self.positional_encodings(x)
       
        return self.dropout(x)
    
# ----- Target masking ----- #  
def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return subsequent_mask == 0


# ----- Transformer ----- #
class TransformerForecaster(nn.Module):
    def __init__(self,  embed_dim, dropout, num_heads, num_layers, hidden_dim, n_inputs=6, n_outputs=1):
        super().__init__()
        # Embedding layers for input and target sequences
        self.input_embed = nn.Linear(n_inputs, embed_dim)
        self.target_embed = nn.Linear(n_outputs, embed_dim)
        
        # Encoder
        self.encoder = Encoder(embed_dim, num_heads, hidden_dim, dropout, num_layers)
        
        # Decoder
        self.decoder = Decoder(embed_dim, num_heads, hidden_dim, dropout, num_layers)
        
        # Output projection
        self.mlp_head = nn.Linear(embed_dim, n_outputs)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, target, src_mask=None):
        """
        Args:
            input: Tensor of shape (batch_size, input_seq_len, n_inputs) - Encoder input
            target: Tensor of shape (batch_size, target_seq_len, n_outputs) - Decoder input
            src_mask: Encoder padding mask (optional)
        Returns:
            Tensor of shape (batch_size, target_seq_len, n_outputs) - Predictions
        """
        # Encoder
        input_embeds = self.input_embed(input)  # Shape: (batch_size, input_seq_len, embed_dim)
        encoder_output = self.encoder(input_embeds, src_mask)  # Shape: (batch_size, input_seq_len, embed_dim)
        
        # Decoder
        target_embeds = self.target_embed(target)  # Shape: (batch_size, target_seq_len, embed_dim)
        tgt_mask = subsequent_mask(target.size(1)).to(target.device)  # Causal mask for decoder
        decoder_output = self.decoder(target_embeds, encoder_output, src_mask, tgt_mask)  # Shape: (batch_size, target_seq_len, embed_dim)
        
        # Project decoder output to the desired output dimension
        output = self.mlp_head(decoder_output)  # Shape: (batch_size, target_seq_len, n_outputs)
        
        return output
    
    
class DecoderForecaster(nn.Module):
    def __init__(self, embed_dim, dropout, num_heads, num_layers, hidden_dim, n_outputs=1):
        super().__init__()
        
        # Target embeddings
        self.target_embed = nn.Linear(n_outputs, embed_dim)
        
        # Decoder
        self.decoder = Decoder(embed_dim, num_heads, hidden_dim, dropout, num_layers)
        
        # Output projection
        self.mlp_head = nn.Linear(embed_dim, n_outputs)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, target, memory, src_mask=None):
        """
        Args:
            target: Tensor of shape (batch_size, target_seq_len, n_outputs) - Decoder input
            memory: Tensor of shape (batch_size, input_seq_len, embed_dim) - Simulated encoder output
            src_mask: Source padding mask (optional)
        Returns:
            Tensor of shape (batch_size, target_seq_len, n_outputs) - Predictions
        """
        # Target embeddings
        target_embeds = self.target_embed(target)  # Shape: (batch_size, target_seq_len, embed_dim)
        
        # Causal mask for target
        tgt_mask = subsequent_mask(target.size(1)).to(target.device)  # Shape: (target_seq_len, target_seq_len)
        
        # Decoder
        decoder_output = self.decoder(target_embeds, memory, src_mask, tgt_mask)  # Shape: (batch_size, target_seq_len, embed_dim)
        
        # Project decoder output to desired output dimension
        output = self.mlp_head(decoder_output)  # Shape: (batch_size, target_seq_len, n_outputs)
        
        return output