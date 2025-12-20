import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


def setup_logging(log_level: int = logging.INFO, log_file: Optional[str] = None):
    """
    Setup logging configuration.

    Args:
        log_level: Logging level
        log_file: Optional log file path
    """
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Reduce verbosity of some third-party loggers
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Optional[Any],
    epoch: int,
    loss: float,
    checkpoint_dir: Union[str, Path],
    filename: Optional[str] = None,
    is_best: bool = False,
    extra_data: Optional[Dict] = None
) -> str:
    """
    Save model checkpoint.

    Args:
        model: PyTorch model
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        epoch: Current epoch
        loss: Current loss value
        checkpoint_dir: Directory to save checkpoint
        filename: Optional custom filename
        is_best: Whether this is the best checkpoint
        extra_data: Additional data to save

    Returns:
        Path to saved checkpoint
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Prepare checkpoint data
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': time.time()
    }

    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    if extra_data:
        checkpoint.update(extra_data)

    # Determine filename
    if filename is None:
        filename = f"checkpoint_epoch_{epoch}.pth"

    checkpoint_path = checkpoint_dir / filename

    # Save checkpoint
    torch.save(checkpoint, checkpoint_path)

    # Save as best if specified
    if is_best:
        best_path = checkpoint_dir / "best_checkpoint.pth"
        torch.save(checkpoint, best_path)

    return str(checkpoint_path)


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Load model checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        model: PyTorch model
        optimizer: Optional optimizer
        scheduler: Optional scheduler
        device: Device to load checkpoint on

    Returns:
        Checkpoint data dictionary
    """
    if device is None:
        device = torch.device('cpu')

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])

    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Load scheduler state if provided
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return checkpoint


def create_optimizer(model: nn.Module, cfg) -> optim.Optimizer:
    """
    Create optimizer based on configuration.

    Args:
        model: PyTorch model
        cfg: Configuration object

    Returns:
        Optimizer instance
    """
    # Separate parameters by learning rates
    decoder_params = []
    backbone_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if any(backbone_name in name for backbone_name in ['vision_encoder', 'text_encoder']):
            backbone_params.append(param)
        else:
            decoder_params.append(param)

    # Create parameter groups
    param_groups = [
        {
            'params': decoder_params,
            'lr': cfg.train.lr_decoder,
            'weight_decay': cfg.train.weight_decay
        }
    ]

    if backbone_params:
        param_groups.append({
            'params': backbone_params,
            'lr': cfg.train.lr_backbone,
            'weight_decay': cfg.train.weight_decay
        })

    # Create optimizer
    optimizer = optim.AdamW(
        param_groups,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    return optimizer


def create_scheduler(
    optimizer: optim.Optimizer,
    cfg,
    steps_per_epoch: int
) -> Optional[Any]:
    """
    Create learning rate scheduler.

    Args:
        optimizer: Optimizer
        cfg: Configuration object
        steps_per_epoch: Number of steps per epoch

    Returns:
        Scheduler instance
    """
    total_steps = cfg.train.epochs * steps_per_epoch
    warmup_steps = cfg.train.warmup_steps

    if warmup_steps > 0:
        # Warmup + Cosine annealing
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps
        )

        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=1e-6
        )

        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )
    else:
        # Just cosine annealing
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=1e-6
        )

    return scheduler


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self, name: str = "", fmt: str = ":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def save_config(config_dict: Dict[str, Any], save_path: Union[str, Path]):
    """
    Save configuration to JSON file.

    Args:
        config_dict: Configuration dictionary
        save_path: Path to save file
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert any non-serializable objects to strings
    serializable_config = {}
    for key, value in config_dict.items():
        try:
            json.dumps(value)
            serializable_config[key] = value
        except (TypeError, ValueError):
            serializable_config[key] = str(value)

    with open(save_path, 'w') as f:
        json.dump(serializable_config, f, indent=2)


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from JSON file.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def create_run_directory(base_dir: Union[str, Path], run_name: Optional[str] = None) -> Path:
    """
    Create a unique run directory with timestamp.

    Args:
        base_dir: Base directory for runs
        run_name: Optional run name

    Returns:
        Path to created run directory
    """
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    # Create timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    if run_name:
        dir_name = f"{run_name}_{timestamp}"
    else:
        dir_name = f"run_{timestamp}"

    run_dir = base_dir / dir_name
    run_dir.mkdir(parents=True, exist_ok=True)

    return run_dir

