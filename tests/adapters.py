from __future__ import annotations

import os
from typing import IO, Any, BinaryIO
from collections.abc import Iterable
from jaxtyping import Float, Int

import numpy.typing as npt
import torch
from torch import Tensor


def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        in_dim (int): The size of the input dimension
        out_dim (int): The size of the output dimension
        weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to

    Returns:
        Float[Tensor, "... d_out"]: The transformed output of your linear module.
    """
    # Import the Linear class from cs336_basics
    from cs336_basics import Linear
    
    # Create a Linear module with the given dimensions
    linear_module = Linear(in_features=d_in, out_features=d_out)
    
    # Load the provided weights into the module
    # The weights tensor has shape (d_out, d_in) which matches our W parameter
    state_dict = {"W": weights}
    linear_module.load_state_dict(state_dict)
    
    # Apply the linear transformation
    with torch.no_grad():
        output = linear_module(in_features)
    
    return output


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.

    Args:
        vocab_size (int): The number of embeddings in the vocabulary
        d_model (int): The size of the embedding dimension
        weights (Float[Tensor, "vocab_size d_model"]): The embedding vectors to fetch from
        token_ids (Int[Tensor, "..."]): The set of token ids to fetch from the Embedding layer

    Returns:
        Float[Tensor, "... d_model"]: Batch of embeddings returned by your Embedding layer.
    """
    # Import the Embedding class from cs336_basics
    from cs336_basics import Embedding
    
    # Create an Embedding module with the given dimensions
    embedding_module = Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
    
    # Load the provided weights into the module
    # The weights tensor has shape (vocab_size, d_model) which matches our weight parameter
    state_dict = {"weight": weights}
    embedding_module.load_state_dict(state_dict)
    
    # Apply the embedding lookup
    with torch.no_grad():
        output = embedding_module(token_ids)
    
    return output


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a SwiGLU network, return
    the output of your implementation with these weights.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W1
        w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for W2
        w3_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W3
        in_features (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.

    Returns:
        Float[Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings.
    """
    # Create SwiGLU module
    from cs336_basics import SwiGLU
    
    swiglu = SwiGLU(d_model=d_model)
    
    # Manually assign the weights
    # Note: The Linear layers store weights with shape (out_features, in_features)
    # but the test provides weights in different orientations
    swiglu.W1.W.data = w1_weight  # w1_weight is (d_ff, d_model)
    swiglu.W2.W.data = w2_weight # w2_weight is (d_model, d_ff), need transpose to (d_ff, d_model)
    swiglu.W3.W.data = w3_weight  # w3_weight is (d_ff, d_model)
    
    # Run forward pass
    return swiglu(in_features)


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Float[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor n x dk
        K (Float[Tensor, " ... keys d_k"]): Key tensor m x dk
        V (Float[Tensor, " ... values d_v"]): Values tensor m x dv
        mask (Float[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    dk = Q.shape[-1]
    # print("Q shape:", Q.shape, "K shape:", K.shape, "V shape:", V.shape)
    QK_T = Q @ torch.swapaxes(K, -1, -2)
    scaled_scores = QK_T / (dk ** 0.5)
    if mask is not None:
        scaled_scores = scaled_scores.masked_fill(mask == 0, float('-inf'))
    attn_weights = torch.softmax(scaled_scores, dim=-1)
    output = attn_weights @ V
    return output

    


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    # Import the MultiHeadSelfAttention class from cs336_basics
    from cs336_basics import MultiHeadSelfAttention
    
    # Create the multi-head self-attention module
    mha = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads)
    
    # The test provides individual head weights, but our implementation uses batched weights
    # We need to construct the full weight matrices for Q, K, V projections
    # Each head has weights of shape (d_k, d_in), and we have num_heads of them
    d_k = q_proj_weight.shape[0]
    d_in = q_proj_weight.shape[1]
    
    # Stack all head weights to create full projection matrices
    # For Q, K, V: repeat the same weights for all heads (since test gives same weights per head)
    # full_q_weight = q_proj_weight.repeat(num_heads, 1)  # (d_model, d_in)
    # full_k_weight = k_proj_weight.repeat(num_heads, 1)  # (d_model, d_in)
    # full_v_weight = v_proj_weight.repeat(num_heads, 1)  # (d_model, d_in)
    
    # Load the weights into the module
    # Our Linear layers expect weights of shape (out_features, in_features)
    state_dict = {
        "W_q.W": q_proj_weight,
        "W_k.W": k_proj_weight,
        "W_v.W": v_proj_weight,
        "W_o.W": o_proj_weight
    }
    # Load state dict with strict=False to be safe (no RoPE in this version)
    mha.load_state_dict(state_dict, strict=False)
    
    # Run forward pass
    with torch.no_grad():
        output = mha(in_features)
    
    return output


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This version of MHA should include RoPE.
    In this case, the RoPE embedding dimension must be the head embedding dimension (d_model // num_heads).
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.
        token_positions (Int[Tensor, " ... sequence_length"] | None): Optional tensor with the positions of the tokens

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    # Import the MultiHeadSelfAttention class from cs336_basics
    from cs336_basics import MultiHeadSelfAttention
    
    # Create the multi-head self-attention module with RoPE support
    mha = MultiHeadSelfAttention(
        d_model=d_model, 
        num_heads=num_heads,
        theta=theta,
        max_seq_len=max_seq_len
    )
    
    # The test provides individual head weights, but our implementation uses batched weights
    # We need to construct the full weight matrices for Q, K, V projections
    # Each head has weights of shape (d_k, d_in), and we have num_heads of them
    d_k = q_proj_weight.shape[0]
    d_in = q_proj_weight.shape[1]
    
    # Load the weights into the module
    # Our Linear layers expect weights of shape (out_features, in_features)
    state_dict = {
        "W_q.W": q_proj_weight,
        "W_k.W": k_proj_weight,
        "W_v.W": v_proj_weight,
        "W_o.W": o_proj_weight
    }
    # Load state dict with strict=False to allow missing RoPE buffers (they'll be initialized automatically)
    mha.load_state_dict(state_dict, strict=False)
    
    # Run forward pass (RoPE is applied automatically inside the forward method)
    with torch.no_grad():
        output = mha(in_features)
    
    return output


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    Run RoPE for a given input tensor.

    Args:
        d_k (int): Embedding dimension size for the query or key tensor.
        theta (float): RoPE parameter.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
        token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
    Returns:
        Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
    """
    # Import the RotaryPositionalEmbedding class from cs336_basics
    from cs336_basics import RotaryPositionalEmbedding
    
    # Create a RoPE module with the given parameters
    rope_module = RotaryPositionalEmbedding(
        theta=theta,
        d_k=d_k,
        max_seq_len=max_seq_len,
        device=in_query_or_key.device
    )
    
    # Apply RoPE to the input tensor
    with torch.no_grad():
        output = rope_module(in_query_or_key, token_positions)
    
    return output


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use RoPE.
    Depending on your implementation, you may simply need to pass the relevant args
    to your TransformerBlock constructor, or you may need to initialize your own RoPE
    class and pass that instead.

    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """
    # Import the TransformerBlock class from cs336_basics
    from cs336_basics import TransformerBlock
    
    # Create the Transformer block with RoPE support
    transformer_block = TransformerBlock(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        theta=theta,
        max_seq_len=max_seq_len,
        device=in_features.device
    )
    
    # Map the test weights to our module's expected state dict format
    state_dict = {
        # Attention weights
        "attention.W_q.W": weights["attn.q_proj.weight"],
        "attention.W_k.W": weights["attn.k_proj.weight"], 
        "attention.W_v.W": weights["attn.v_proj.weight"],
        "attention.W_o.W": weights["attn.output_proj.weight"],
        
        # RMSNorm weights
        "norm1.weight": weights["ln1.weight"],
        "norm2.weight": weights["ln2.weight"],
        
        # Feed-forward weights
        "feed_forward.W1.W": weights["ffn.w1.weight"],  # Transpose from (d_model, d_ff) to (d_ff, d_model)
        "feed_forward.W2.W": weights["ffn.w2.weight"],   # Already (d_ff, d_model)
        "feed_forward.W3.W": weights["ffn.w3.weight"],  # Transpose from (d_model, d_ff) to (d_ff, d_model)
    }
    
    # Load state dict with strict=False to allow missing RoPE buffers (they'll be initialized automatically)
    transformer_block.load_state_dict(state_dict, strict=False)
    
    # Run forward pass
    with torch.no_grad():
        output = transformer_block(in_features)
    
    return output


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    This function should use RoPE.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE $Theta$ parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
    """
    # Import the TransformerLM class from cs336_basics
    from cs336_basics import TransformerLM
    
    # Create the Transformer language model with RoPE
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        theta=rope_theta,
        max_seq_len=context_length
    )
    
    # Build the state dict mapping from the provided weights to our model's structure
    state_dict = {}
    
    # Token embeddings
    state_dict["token_embedding.weight"] = weights["token_embeddings.weight"]
    
    # Layer weights
    for layer_idx in range(num_layers):
        # Attention weights
        state_dict[f"blocks.{layer_idx}.attention.W_q.W"] = weights[f"layers.{layer_idx}.attn.q_proj.weight"]
        state_dict[f"blocks.{layer_idx}.attention.W_k.W"] = weights[f"layers.{layer_idx}.attn.k_proj.weight"]
        state_dict[f"blocks.{layer_idx}.attention.W_v.W"] = weights[f"layers.{layer_idx}.attn.v_proj.weight"]
        state_dict[f"blocks.{layer_idx}.attention.W_o.W"] = weights[f"layers.{layer_idx}.attn.output_proj.weight"]
        
        # Layer norm weights
        state_dict[f"blocks.{layer_idx}.norm1.weight"] = weights[f"layers.{layer_idx}.ln1.weight"]
        state_dict[f"blocks.{layer_idx}.norm2.weight"] = weights[f"layers.{layer_idx}.ln2.weight"]
        
        # Feed-forward weights
        state_dict[f"blocks.{layer_idx}.feed_forward.W1.W"] = weights[f"layers.{layer_idx}.ffn.w1.weight"]
        state_dict[f"blocks.{layer_idx}.feed_forward.W2.W"] = weights[f"layers.{layer_idx}.ffn.w2.weight"]
        state_dict[f"blocks.{layer_idx}.feed_forward.W3.W"] = weights[f"layers.{layer_idx}.ffn.w3.weight"]
    
    # Final layer norm and output projection
    state_dict["final_norm.weight"] = weights["ln_final.weight"]
    state_dict["lm_head.W"] = weights["lm_head.weight"]
    
    # Load the state dict (with strict=False to handle RoPE buffers)
    model.load_state_dict(state_dict, strict=False)
    
    # Run forward pass
    with torch.no_grad():
        output = model(in_indices)
    
    return output


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.

    Args:
        d_model (int): The dimensionality of the RMSNorm input.
        eps: (float): A value added to the denominator for numerical stability.
        weights (Float[Tensor, "d_model"]): RMSNorm weights.
        in_features (Float[Tensor, "... d_model"]): Input features to run RMSNorm on. Can have arbitrary leading
            dimensions.

    Returns:
        Float[Tensor,"... d_model"]: Tensor of with the same shape as `in_features` with the output of running
        RMSNorm of the `in_features`.
    """
    # Import the RMSNorm class from cs336_basics
    from cs336_basics import RMSNorm
    
    # Create an RMSNorm module with the given parameters
    rmsnorm_module = RMSNorm(d_model=d_model, eps=eps)
    
    # Load the provided weights into the module
    state_dict = {"weight": weights}
    rmsnorm_module.load_state_dict(state_dict)
    
    # Apply RMSNorm to the input features
    with torch.no_grad():
        output = rmsnorm_module(in_features)
    
    return output


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """Given a tensor of inputs, return the output of applying SiLU
    to each element.

    Args:
        in_features(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

    Returns:
        Float[Tensor,"..."]: of with the same shape as `in_features` with the output of applying
        SiLU to each element.
    """
    raise NotImplementedError


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    # Import the get_batch function from cs336_basics
    from cs336_basics.dataloader_utlis import get_batch
    
    # Call our implementation and return the result
    return get_batch(dataset, batch_size, context_length, device)


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    return torch.softmax(in_features, dim=dim)



def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    # Import the cross_entropy function from cs336_basics
    from cs336_basics import cross_entropy
    
    # Compute and return the cross entropy loss
    return cross_entropy(inputs, targets)


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    # Import the gradient_clipping function from cs336_basics
    from cs336_basics import gradient_clipping
    
    # Apply gradient clipping
    gradient_clipping(parameters, max_l2_norm)


def get_adamw_cls() -> Any:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    # Import the AdamW class from cs336_basics
    from cs336_basics.train_utilis import AdamW
    
    return AdamW


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    # Import the cosine learning rate schedule function from cs336_basics
    from cs336_basics import cosine_lr_schedule
    
    # Call our implementation with the appropriate parameters
    return cosine_lr_schedule(
        t=it,
        alpha_max=max_learning_rate,
        alpha_min=min_learning_rate,
        T_w=warmup_iters,
        T_c=cosine_cycle_iters
    )


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    # Import the save_checkpoint function from cs336_basics
    from cs336_basics import save_checkpoint
    
    # Call our implementation
    save_checkpoint(model, optimizer, iteration, out)


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    # Import the load_checkpoint function from cs336_basics
    from cs336_basics import load_checkpoint
    
    # Call our implementation and return the iteration number
    return load_checkpoint(src, model, optimizer)


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    raise NotImplementedError


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    # from cs336_basics.tokenizer_utils import train_bpe_efficient
    # return train_bpe_efficient(str(input_path), vocab_size, special_tokens)
    from cs336_basics.tokenizer_utils import train_bpe
    return train_bpe(str(input_path), vocab_size, special_tokens)
