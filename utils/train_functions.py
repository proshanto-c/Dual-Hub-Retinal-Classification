import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from typing import Any, Dict, List, Optional
from utils.metrics import evaluate_multilabel_predictions
from utils.util_functions import autocast_context

# Assuming these are imported at the top of your train.py file
# from util_functions import autocast_context
# from metrics import evaluate_multilabel_predictions

def graph_epoch(
    model: nn.Module,
    loader: Any,
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool,
    label_columns: List[str],
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Any = None,
    desc: str = "Processing",
    threshold: float = 0.5,
    accum_steps: int = 1
) -> Dict[str, Any]:
    """
    Executes a single epoch (training or validation) for the Heterogeneous Graph Neural Network.
    If an optimizer is provided, the model will backpropagate and update weights.
    """
    is_training = optimizer is not None
    model.train(is_training)
    
    total_loss = 0.0
    total_examples = 0
    all_targets = []
    all_probs = []

    # Wrap the loader in a progress bar
    pbar = tqdm(loader, desc=desc, leave=False)
    
    if is_training and optimizer is not None and (0 % accum_steps == 0):
        optimizer.zero_grad(set_to_none=True)
    
    for i, batch in enumerate(pbar):
        batch = batch.to(device)
        
        with torch.set_grad_enabled(is_training):
            # 1. Forward Pass using modular autocast context
            with autocast_context(amp_enabled):
                logits = model(batch)
                # Reshape batch.y to match logits dynamically
                loss = criterion(logits, batch.y.view_as(logits).float())
                if is_training and accum_steps > 1:
                    loss_scaled = loss / accum_steps
                else:
                    loss_scaled = loss

            # 2. Backward Pass (Training Only)
            if is_training:
                if scaler is not None and amp_enabled:
                    scaler.scale(loss_scaled).backward()
                    if (i + 1) % accum_steps == 0 or (i + 1) == len(loader):
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                else:
                    loss_scaled.backward()
                    if (i + 1) % accum_steps == 0 or (i + 1) == len(loader):
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)

        # 3. Batch Tracking
        bs = int(batch.y.size(0))
        current_loss = float(loss.item())
        total_loss += current_loss * bs
        total_examples += bs
        
        pbar.set_postfix({"loss": f"{current_loss:.4f}"})
        
        all_targets.append(batch.y.detach().cpu().numpy())
        all_probs.append(torch.sigmoid(logits).detach().cpu().numpy())

    # 4. Aggregation
    targets = np.concatenate(all_targets, axis=0)
    probabilities = np.concatenate(all_probs, axis=0)
    
    # Ensure 2D shapes for metrics calculation
    if targets.ndim == 1 and probabilities.ndim > 1:
        targets = targets.reshape(-1, probabilities.shape[1])
        
    # 5. Metric Evaluation via modular metrics.py
    metrics_frame, summary, _ = evaluate_multilabel_predictions(
        targets=targets, 
        probabilities=probabilities, 
        label_names=label_columns, 
        thresholds=threshold
    )
    
    return {
        "loss": total_loss / max(total_examples, 1),
        "metrics_frame": metrics_frame,
        "summary": summary,
        "targets": targets,
        "probabilities": probabilities,
    }

def image_epoch(
    model: nn.Module,
    loader: Any,
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool,
    label_columns: List[str],
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Any = None,
    desc: str = "Processing",
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Executes a single epoch (training or validation) for standard image models.
    """
    is_training = optimizer is not None
    model.train(is_training)
    
    total_loss = 0.0
    total_examples = 0
    all_targets = []
    all_probs = []

    # Wrap the loader in a progress bar
    pbar = tqdm(loader, desc=desc, leave=False)
    
    for i, batch in enumerate(pbar):
        # image dataset returns dict with "image" and "target"
        images = batch["image"].to(device)
        targets = batch["target"].to(device)
        
        with torch.set_grad_enabled(is_training):
            # 1. Forward Pass
            with autocast_context(amp_enabled):
                if "context" in batch:
                    contexts = batch["context"].to(device)
                    logits = model(images, contexts)
                else:
                    logits = model(images)
                loss = criterion(logits, targets.view_as(logits).float())

            # 2. Backward Pass
            if is_training:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and amp_enabled:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        # 3. Batch Tracking
        bs = int(targets.size(0))
        current_loss = float(loss.item())
        total_loss += current_loss * bs
        total_examples += bs
        
        pbar.set_postfix({"loss": f"{current_loss:.4f}"})
        
        all_targets.append(targets.detach().cpu().numpy())
        all_probs.append(torch.sigmoid(logits).detach().cpu().numpy())

    # 4. Aggregation
    targets_np = np.concatenate(all_targets, axis=0)
    probabilities_np = np.concatenate(all_probs, axis=0)
    
    # Ensure 2D shapes for metrics calculation
    if targets_np.ndim == 1 and probabilities_np.ndim > 1:
        targets_np = targets_np.reshape(-1, probabilities_np.shape[1])
        
    # 5. Metric Evaluation
    metrics_frame, summary, _ = evaluate_multilabel_predictions(
        targets=targets_np, 
        probabilities=probabilities_np, 
        label_names=label_columns, 
        thresholds=threshold
    )
    
    return {
        "loss": total_loss / max(total_examples, 1),
        "metrics_frame": metrics_frame,
        "summary": summary,
        "targets": targets_np,
        "probabilities": probabilities_np,
    }