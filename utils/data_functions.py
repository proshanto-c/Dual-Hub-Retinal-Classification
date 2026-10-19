import random
import numpy as np
import torch
from torch.utils.data import Dataset

def calculate_positive_weights(targets: np.ndarray) -> torch.Tensor:
    """
    Calculates positive weights for BCEWithLogitsLoss to handle multilabel class imbalance.
    """
    positives = targets.sum(axis=0).astype(np.float32)
    total_samples = targets.shape[0]
    negatives = total_samples - positives

    weights = np.divide(negatives, np.maximum(positives, 1.0), dtype=np.float32)
    weights[~np.isfinite(weights)] = 1.0
    weights[positives == 0] = 1.0
    
    # Dampen the weights using square root to prevent extreme clumping of probabilities
    weights = np.sqrt(weights)

    return torch.tensor(weights, dtype=torch.float32)


def worker_init_fn(worker_id: int) -> None:
    """Ensures diverse random seeding across DataLoader workers."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    seed: int,
    prefetch_factor: int = 2,
    is_graph: bool = False
):
    """
    A standardized DataLoader constructor. Automatically routes to the PyTorch Geometric
    DataLoader if is_graph=True, allowing HeteroData objects to be batched correctly.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": False,
        "worker_init_fn": worker_init_fn,
        "generator": generator,
    }
    
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = prefetch_factor
        
    if is_graph:
        kwargs["pin_memory"] = True
        if num_workers > 0:
            kwargs["persistent_workers"] = True
            kwargs["prefetch_factor"] = prefetch_factor if prefetch_factor is not None else 3
        else:
            kwargs["persistent_workers"] = False
            kwargs["prefetch_factor"] = None

        from torch_geometric.loader import DataLoader as PyGDataLoader
        return PyGDataLoader(**kwargs)
    else:
        from torch.utils.data import DataLoader as StandardDataLoader
        return StandardDataLoader(**kwargs)