"""
Neural network utility modules for CS336 Assignment 1.
"""

import torch
import torch.nn as nn
import torch.nn.init as init
from jaxtyping import Float
from torch import Tensor
from typing import Optional



class Linear(nn.Module):
    """
    Linear transformation module that performs matrix multiplication without bias.
    
    This module implements a linear transformation y = xW^T where W is the weight matrix.
    The weight matrix W has shape (out_features, in_features) for memory ordering reasons.
    """
    
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        device: torch.device | None = None, 
        dtype: torch.dtype | None = None
    ):
        """
        Construct a linear transformation module.
        
        Args:
            in_features: Final dimension of the input
            out_features: Final dimension of the output  
            device: Device to store the parameters on
            dtype: Data type of the parameters
        """
        super().__init__()
        
        # Store the feature dimensions
        self.in_features = in_features
        self.out_features = out_features
        
        # Create weight parameter with shape (out_features, in_features)
        # This is W (not W^T) for memory ordering reasons
        self.W = nn.Parameter(
            torch.empty(
                (out_features, in_features), 
                device=device, 
                dtype=dtype
            )
        )
        
        # Initialize weights using truncated normal distribution
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        """Initialize the weight parameter using truncated normal distribution."""
        init.trunc_normal_(self.W)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply the linear transformation to the input.
        
        Args:
            x: Input tensor with last dimension equal to in_features
            
        Returns:
            Output tensor with last dimension equal to out_features
        """
        # Perform matrix multiplication: x @ W^T
        # Since W has shape (out_features, in_features), W.T has shape (in_features, out_features)
        # So x @ W.T gives us the correct output shape
        # print("w shape:", self.W.shape)
        return x @ self.W.T



class Embedding(nn.Module):
    """Embedding layer that performs embedding lookup."""
    
    def __init__(
        self, 
        num_embeddings: int, 
        embedding_dim: int, 
        device: Optional[torch.device] = None, 
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()
        
        # Store dimensions
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        # Create the embedding weight matrix as a parameter
        # Shape: (num_embeddings, embedding_dim)
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        
        # Initialize weights using truncated normal distribution
        torch.nn.init.trunc_normal_(self.weight)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Lookup embedding vectors for the given token IDs.
        
        Args:
            token_ids: Tensor of token IDs, shape (...,)
            
        Returns:
            Embedding vectors, shape (..., embedding_dim)
        """
        # Use advanced indexing to lookup embeddings
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        """
        Construct the RMSNorm module.
        
        Args:
            d_model: Hidden dimension of the model
            eps: Epsilon value for numerical stability
            device: Device to store the parameters on
            dtype: Data type of the parameters
        """
        super().__init__()
        
        self.eps = eps
        self.d_model = d_model
        
        # Create learnable scale parameter (gamma)
        self.weight = nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor and return a tensor of the same shape.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, d_model)
            
        Returns:
            Normalized tensor of the same shape as input
        """
        # Store original dtype for later downcast
        original_dtype = x.dtype
        
        # Upcast to float32 for numerical stability
        x = x.to(torch.float32)
        
        # Compute RMS: sqrt(mean(x^2))
        # Keep dimension for broadcasting
        rms = torch.sqrt(torch.mean(x.square(), dim=-1, keepdim=True) + self.eps)
        
        # Normalize by RMS
        x_normalized = x / rms
        
        # Scale by learnable parameter
        x_scaled = x_normalized * self.weight
        
        # Downcast back to original dtype
        return x_scaled.to(original_dtype)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network with SiLU activation and GLU."""
    
    def __init__(
        self,
        d_model: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        """
        Construct the SwiGLU module.
        
        Args:
            d_model: Hidden dimension of the model
            device: Device to store the parameters on
            dtype: Data type of the parameters
        """
        super().__init__()
        
        self.d_model = d_model
        
        # Calculate d_ff as approximately 8/3 * d_model, rounded to nearest multiple of 64
        d_ff = int((8/3) * d_model)
        # Round to nearest multiple of 64 for hardware efficiency
        d_ff = ((d_ff ) // 64) * 64
        self.d_ff = d_ff
        # Linear projections
        self.W1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.W2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.W3 = Linear(d_model, d_ff, device=device, dtype=dtype)

        # print("W1 shape:", self.W1.W.shape)  # (d_ff, d_model)
        # print("W2 shape:", self.W2.W.shape)  # (d_model, d_ff)
        # print("W3 shape:", self.W3.W.shape)  # (d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SwiGLU transformation to input.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, d_model)
            
        Returns:
            Output tensor of shape (batch_size, sequence_length, d_model)
        """
        # SwiGLU(x, W1, W2, W3) = (SiLU(xW1) ⊙ xW2)W3
        # where ⊙ is element-wise multiplication (Hadamard product)
        
        # Apply first linear transformation and SiLU activation
        x1 = self.W1(x)
        silu_x1 = x1 * torch.sigmoid(x1)        
        x3 = self.W3(x)
        gated = silu_x1 * x3
        output = self.W2(gated)
        return output


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Positional Embedding (RoPE) module."""
    
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None
    ):
        """
        Construct the RoPE module and create buffers if needed.
        
        Args:
            theta: Θ value for the RoPE
            d_k: dimension of query and key vectors
            max_seq_len: Maximum sequence length that will be inputted
            device: Device to store the buffer on
        """
        super().__init__()
        
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        
        # Precompute frequency values
        # freq_i = 1 / (theta^(2i/d_k)) for i = 0, 1, ..., d_k/2 - 1
        i = torch.arange(0, d_k // 2, dtype=torch.float32, device=device)
        freqs = 1.0 / (theta ** (2 * i / d_k))
        
        # Precompute position encodings for all possible positions
        # positions: [0, 1, 2, ..., max_seq_len - 1]
        positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)
        
        # Create the angle matrix: positions * freqs
        # Shape: (max_seq_len, d_k // 2)
        angles = torch.outer(positions, freqs)
        
        # Precompute cos and sin values
        # Shape: (max_seq_len, d_k // 2)
        cos_vals = torch.cos(angles)
        sin_vals = torch.sin(angles)
        
        # Register as buffers so they move with the module but aren't parameters
        self.register_buffer('cos_vals', cos_vals)
        self.register_buffer('sin_vals', sin_vals)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor and return a tensor of the same shape.
        
        Args:
            x: Input tensor of shape (..., seq_len, d_k)
            token_positions: Tensor of shape (..., seq_len) specifying token positions
            
        Returns:
            Rotated tensor of the same shape as input
        """
        # Get the sequence length and ensure d_k is even
        *batch_dims, seq_len, d_k = x.shape
        assert d_k == self.d_k, f"Input d_k ({d_k}) doesn't match initialized d_k ({self.d_k})"
        assert d_k % 2 == 0, f"d_k must be even, got {d_k}"
        
        # Reshape x to split into two halves for rotation
        # Shape: (..., seq_len, d_k // 2, 2)
        x_reshaped = x.view(*batch_dims, seq_len, d_k // 2, 2)
        x1 = x_reshaped[..., 0]  # First half: (..., seq_len, d_k // 2)
        x2 = x_reshaped[..., 1]  # Second half: (..., seq_len, d_k // 2)
        
        # Get cos and sin values for the given positions
        # token_positions shape: (..., seq_len)
        # We need to index into our precomputed cos_vals and sin_vals
        cos = self.cos_vals[token_positions]  # (..., seq_len, d_k // 2)
        sin = self.sin_vals[token_positions]  # (..., seq_len, d_k // 2)
        
        # Apply rotation: [cos -sin; sin cos] @ [x1; x2]
        # x1_rot = x1 * cos - x2 * sin
        # x2_rot = x1 * sin + x2 * cos
        x1_rot = x1 * cos - x2 * sin
        x2_rot = x1 * sin + x2 * cos
        
        # Recombine the rotated halves
        # Shape: (..., seq_len, d_k // 2, 2)
        x_rot_reshaped = torch.stack([x1_rot, x2_rot], dim=-1)
        
        # Reshape back to original shape: (..., seq_len, d_k)
        x_rot = x_rot_reshaped.view(*batch_dims, seq_len, d_k)
        
        return x_rot


class MultiHeadSelfAttention(nn.Module):
    """Causal multi-head self-attention."""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        # RoPE parameters (optional)
        theta: float | None = None,
        max_seq_len: int | None = None
    ):
        """
        Construct the multi-head self-attention module.
        
        Args:
            d_model: Dimensionality of the Transformer block inputs
            num_heads: Number of heads to use in multi-head self-attention
            device: Device to store the parameters on
            dtype: Data type of the parameters
            theta: RoPE theta parameter (if None, RoPE is not used)
            max_seq_len: Maximum sequence length for RoPE (required if theta is provided)
        """
        super().__init__()
        
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # d_k = d_v = d_model / h
        self.d_v = self.d_k
        
        # Linear projections for Q, K, V
        self.W_q = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_k = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_v = Linear(d_model, d_model, device=device, dtype=dtype)
        
        # Output projection
        self.W_o = Linear(d_model, d_model, device=device, dtype=dtype)
        
        # RoPE support
        self.use_rope = theta is not None
        if self.use_rope:
            assert max_seq_len is not None, "max_seq_len must be provided when using RoPE"
            self.rope = RotaryPositionalEmbedding(
                theta=theta,
                d_k=self.d_k,
                max_seq_len=max_seq_len,
                device=device
            )
        else:
            self.rope = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply causal multi-head self-attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        
        # Linear projections
        Q = self.W_q(x)  # (batch_size, seq_len, d_model)
        K = self.W_k(x)  # (batch_size, seq_len, d_model)
        V = self.W_v(x)  # (batch_size, seq_len, d_model)
        
        # Reshape for multi-head attention
        # Split d_model into num_heads * d_k
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)  # (batch_size, num_heads, seq_len, d_k)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)  # (batch_size, num_heads, seq_len, d_k)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_v).transpose(1, 2)  # (batch_size, num_heads, seq_len, d_v)
        
        # Apply RoPE to Q and K (if enabled)
        if self.use_rope:
            # Create token positions for each sequence position
            token_positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)  # (batch_size, seq_len)
            
            # Apply RoPE to Q and K
            # RoPE expects shape (..., seq_len, d_k), so we need to handle the head dimension properly
            # We'll apply RoPE to each head separately by treating num_heads as part of the batch dimension
            batch_heads, seq_len_q, d_k = Q.shape[0] * Q.shape[1], Q.shape[2], Q.shape[3]
            
            # Reshape to treat head dimension as batch dimension for RoPE
            Q_reshaped = Q.contiguous().view(batch_heads, seq_len_q, d_k)  # (batch_size * num_heads, seq_len, d_k)
            K_reshaped = K.contiguous().view(batch_heads, seq_len_q, d_k)  # (batch_size * num_heads, seq_len, d_k)
            
            # Expand token positions for each head
            token_positions_expanded = token_positions.unsqueeze(1).expand(-1, self.num_heads, -1).contiguous().view(batch_heads, seq_len)  # (batch_size * num_heads, seq_len)
            
            # Apply RoPE
            Q_reshaped = self.rope(Q_reshaped, token_positions_expanded)
            K_reshaped = self.rope(K_reshaped, token_positions_expanded)
            
            # Reshape back to original multi-head format
            Q = Q_reshaped.view(batch_size, self.num_heads, seq_len, self.d_k)
            K = K_reshaped.view(batch_size, self.num_heads, seq_len, self.d_k)
        
        # Apply scaled dot-product attention with causal mask
        attn_output = self._scaled_dot_product_attention(Q, K, V)  # (batch_size, num_heads, seq_len, d_v)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)  # (batch_size, seq_len, d_model)
        
        # Apply output projection
        output = self.W_o(attn_output)  # (batch_size, seq_len, d_model)
        
        return output
    
    def _scaled_dot_product_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """
        Apply scaled dot-product attention with causal masking.
        
        Args:
            Q: Query tensor of shape (batch_size, num_heads, seq_len, d_k)
            K: Key tensor of shape (batch_size, num_heads, seq_len, d_k)
            V: Value tensor of shape (batch_size, num_heads, seq_len, d_v)
            
        Returns:
            Attention output of shape (batch_size, num_heads, seq_len, d_v)
        """
        batch_size, num_heads, seq_len, d_k = Q.shape
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)  # (batch_size, num_heads, seq_len, seq_len)
        
        # Apply causal mask (lower triangular)
        # Create causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len, device=Q.device, dtype=torch.bool))
        
        # Apply mask by setting masked positions to -inf
        scores = scores.masked_fill(~mask, float('-inf'))
        
        # Apply softmax
        attn_weights = torch.softmax(scores, dim=-1)  # (batch_size, num_heads, seq_len, seq_len)
        
        # Apply attention weights to values
        attn_output = torch.matmul(attn_weights, V)  # (batch_size, num_heads, seq_len, d_v)
        
        return attn_output


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block with multi-head self-attention and feed-forward network."""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        # Optional RoPE parameters
        theta: float | None = None,
        max_seq_len: int | None = None,
        eps: float = 1e-5
    ):
        """
        Construct the Transformer block.
        
        Args:
            d_model: Dimensionality of the Transformer block inputs
            num_heads: Number of heads to use in multi-head self-attention
            d_ff: Dimensionality of the position-wise feed-forward inner layer
            device: Device to store parameters on
            dtype: Data type of parameters
            theta: RoPE theta parameter (if None, RoPE is not used)
            max_seq_len: Maximum sequence length for RoPE (required if theta is provided)
            eps: Epsilon value for RMSNorm numerical stability
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        
        # First sublayer: Multi-head self-attention
        self.norm1 = RMSNorm(d_model, eps=eps, device=device, dtype=dtype)
        self.attention = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            device=device,
            dtype=dtype,
            theta=theta,
            max_seq_len=max_seq_len
        )
        
        # Second sublayer: Feed-forward network
        self.norm2 = RMSNorm(d_model, eps=eps, device=device, dtype=dtype)
        self.feed_forward = SwiGLU(d_model, device=device, dtype=dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the Transformer block to input.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # First sublayer: y = x + MultiHeadSelfAttention(RMSNorm(x))
        normed_x1 = self.norm1(x)
        attn_output = self.attention(normed_x1)
        x = x + attn_output  # Residual connection
        
        # Second sublayer: y = x + SwiGLU(RMSNorm(x))
        normed_x2 = self.norm2(x)
        ff_output = self.feed_forward(normed_x2)
        x = x + ff_output  # Residual connection
        
        return x


class TransformerLM(nn.Module):
    """Transformer Language Model."""
    
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        # RoPE parameters (optional)
        theta: float | None = None,
        max_seq_len: int | None = None
    ):
        """
        Construct the Transformer language model.
        
        Args:
            vocab_size: The size of the vocabulary
            context_length: The maximum context length
            d_model: Dimensionality of the model
            num_heads: Number of heads to use in multi-head self-attention
            d_ff: Dimensionality of the position-wise feed-forward inner layer
            num_layers: The number of Transformer blocks to use
            device: Device to store parameters on
            dtype: Data type of parameters
            theta: RoPE theta parameter (if None, use learned position embeddings)
            max_seq_len: Maximum sequence length for RoPE (required if theta is provided)
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.use_rope = theta is not None
        
        # Token embeddings
        self.token_embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        
        # Position embeddings (only if not using RoPE)
        if not self.use_rope:
            self.position_embedding = Embedding(context_length, d_model, device=device, dtype=dtype)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                device=device,
                dtype=dtype,
                theta=theta,
                max_seq_len=max_seq_len
            )
            for _ in range(num_layers)
        ])
        
        # Final layer normalization
        self.final_norm = RMSNorm(d_model, device=device, dtype=dtype)
        
        # Output projection to vocabulary
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Transformer language model.
        
        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            
        Returns:
            Logits of shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        token_embeds = self.token_embedding(input_ids)  # (batch_size, seq_len, d_model)
        
        # Add position embeddings (if not using RoPE)
        if not self.use_rope:
            # Create position indices
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
            position_embeds = self.position_embedding(position_ids)  # (batch_size, seq_len, d_model)
            x = token_embeds + position_embeds
        else:
            x = token_embeds
        
        # Apply Transformer blocks
        for block in self.blocks:
            x = block(x)  # (batch_size, seq_len, d_model)
        
        # Final layer normalization
        x = self.final_norm(x)  # (batch_size, seq_len, d_model)
        
        # Output projection to vocabulary
        logits = self.lm_head(x)  # (batch_size, seq_len, vocab_size)
        
        return logits


