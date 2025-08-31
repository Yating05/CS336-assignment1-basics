import torch
from typing import Optional, Union


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute cross entropy loss with numerical stability.
    
    Args:
        logits: Predicted logits of shape (..., vocab_size)
        targets: Target token indices of shape (...,)
        
    Returns:
        Average cross entropy loss across the batch
    """
    # Get the shape information
    *batch_dims, vocab_size = logits.shape
    batch_size = 1
    for dim in batch_dims:
        batch_size *= dim
    
    # Flatten logits and targets for easier processing
    # logits: (batch_size, vocab_size)
    # targets: (batch_size,)
    logits_flat = logits.view(batch_size, vocab_size)
    targets_flat = targets.view(batch_size)
    
    # Subtract the maximum logit for numerical stability
    # This prevents overflow in the exp operation
    max_logits = torch.max(logits_flat, dim=1, keepdim=True).values  # (batch_size, 1)
    logits_stable = logits_flat - max_logits  # (batch_size, vocab_size)
    
    # Compute log softmax efficiently
    # log_softmax(x_i) = x_i - max(x) - log(sum(exp(x_j - max(x))))
    exp_logits = torch.exp(logits_stable)  # (batch_size, vocab_size)
    sum_exp_logits = torch.sum(exp_logits, dim=1, keepdim=True)  # (batch_size, 1)
    log_sum_exp = torch.log(sum_exp_logits)  # (batch_size, 1)
    
    # Get the logits for the target tokens
    # Use advanced indexing to select the logit for each target
    target_logits = logits_stable[torch.arange(batch_size), targets_flat]  # (batch_size,)
    target_logits = target_logits.unsqueeze(1)  # (batch_size, 1)
    
    # Compute cross entropy: -log_softmax(target)
    # log_softmax(target) = target_logit - log_sum_exp
    log_softmax_target = target_logits - log_sum_exp  # (batch_size, 1)
    cross_entropy_loss = -log_softmax_target.squeeze(1)  # (batch_size,)
    
    # Return the average loss across the batch
    return torch.mean(cross_entropy_loss)

class AdamW(torch.optim.Optimizer):
    """
    AdamW optimizer implementation.
    
    This implements the AdamW algorithm as described in "Decoupled Weight Decay Regularization"
    by Loshchilov & Hutter (2017).
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2
    ):
        """
        Initialize the AdamW optimizer.
        
        Args:
            params: Iterable of parameters to optimize or dicts defining parameter groups
            lr: Learning rate (α)
            betas: Coefficients used for computing running averages of gradient and its square (β₁, β₂)
            eps: Term added to the denominator to improve numerical stability (ε)
            weight_decay: Weight decay coefficient (λ)
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    def step(self, closure=None):
        """
        Perform a single optimization step.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                # Exponential moving average of gradient values
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # Exponential moving average of squared gradient values
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Compute bias-corrected first and second moment estimates
                corrected_exp_avg = exp_avg / bias_correction1
                corrected_exp_avg_sq = exp_avg_sq / bias_correction2
                
                # Update parameters
                # AdamW: p = p - α * (m̂ / (√v̂ + ε) + λ * p)
                # Where the weight decay term is applied directly to the parameters
                
                # Apply weight decay (decoupled from gradient)
                p.data.mul_(1 - group['lr'] * group['weight_decay'])
                
                # Apply Adam update
                denom = corrected_exp_avg_sq.sqrt().add_(group['eps'])
                p.data.addcdiv_(corrected_exp_avg, denom, value=-group['lr'])
        
        return loss


def cosine_lr_schedule(
    t: int,
    alpha_max: float,
    alpha_min: float,
    T_w: int,
    T_c: int
) -> float:
    """
    Cosine learning rate schedule with warmup.
    
    Args:
        t: Current step
        alpha_max: Maximum learning rate (reached after warmup)
        alpha_min: Minimum learning rate (at the end of cosine decay)
        T_w: Number of warmup steps
        T_c: Total number of steps for cosine decay (including warmup)
    
    Returns:
        Learning rate at step t
    """
    import math
    
    if t < T_w:
        # Warmup phase: linear increase from 0 to alpha_max
        return alpha_max * (t / T_w)
    elif t < T_c:
        # Cosine decay phase
        # Progress through cosine decay (0 to 1)
        progress = (t - T_w) / (T_c - T_w)
        # Cosine decay formula: alpha_min + 0.5 * (alpha_max - alpha_min) * (1 + cos(π * progress))
        return alpha_min + 0.5 * (alpha_max - alpha_min) * (1 + math.cos(math.pi * progress))
    else:
        # After cosine decay: maintain minimum learning rate
        return alpha_min


def gradient_clipping(parameters, max_l2_norm, eps=1e-6):
    """
    Clip gradients to have maximum L2 norm.
    
    Args:
        parameters: Iterable of parameters (torch.nn.Parameter objects)
        max_l2_norm: Maximum L2 norm for the gradients
        eps: Small epsilon value for numerical stability (default: 1e-6)
    """
    import torch
    from typing import Iterable
    
    # Convert to list if it's not already
    if not isinstance(parameters, list):
        parameters = list(parameters)
    
    # Compute the total L2 norm of all gradients
    total_norm = 0.0
    for param in parameters:
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)  # L2 norm
            total_norm += param_norm.item() ** 2
    
    total_norm = total_norm ** 0.5  # Square root to get L2 norm
    
    # Compute the clipping coefficient
    clip_coef = max_l2_norm / (total_norm + eps)
    
    # Only clip if the total norm exceeds the maximum
    if clip_coef < 1.0:
        for param in parameters:
            if param.grad is not None:
                param.grad.data.mul_(clip_coef)


def save_checkpoint(model, optimizer, iteration, out):
    """
    Save model, optimizer, and iteration state to a checkpoint.
    
    Args:
        model: torch.nn.Module to save
        optimizer: torch.optim.Optimizer to save
        iteration: int current iteration number
        out: str | os.PathLike | BinaryIO | IO[bytes] output path or file-like object
    """
    import torch
    
    # Create checkpoint dictionary with all necessary state
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }
    
    # Save the checkpoint
    torch.save(checkpoint, out)


def load_checkpoint(src, model, optimizer):
    """
    Load model, optimizer, and iteration state from a checkpoint.
    
    Args:
        src: str | os.PathLike | BinaryIO | IO[bytes] input path or file-like object
        model: torch.nn.Module to load state into
        optimizer: torch.optim.Optimizer to load state into
        
    Returns:
        int: The iteration number from the checkpoint
    """
    import torch
    
    # Load the checkpoint
    checkpoint = torch.load(src)
    
    # Restore model and optimizer states
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Return the iteration number
    return checkpoint['iteration']