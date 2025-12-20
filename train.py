import os

# Set tokenizers parallelism before any other imports
# This prevents the "forked process after parallelism" warning when using DataLoader workers
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import argparse
import logging
import random
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.distributed as dist
import wandb

matplotlib.use('Agg')  # Use non-interactive backend

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Import dataio
from trainer.trainer import TrajectoryDiffusionTrainer
from utils.misc import setup_logging

logger = logging.getLogger(__name__)


def setup_distributed():
    """Initialize distributed training."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    if world_size > 1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def setup_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For reproducibility with CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Trajectory Prediction")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    
    # NEW: Accept arbitrary config overrides with dot notation
    parser.add_argument(
        "--override", 
        nargs="*", 
        default=[],
        help="Override config values using dot notation (e.g., train.batch_size=64 hardware.device=cuda:2)"
    )

    args = parser.parse_args()

    # ADD: Setup distributed training
    rank, world_size, local_rank = setup_distributed()

    # Setup logging early (only on main process)
    if rank == 0:
        log_level = logging.DEBUG if args.debug else logging.INFO
        setup_logging(log_level)

    # Load configuration with OmegaConf
    from omegaconf import OmegaConf
    
    # Load base config
    cfg = OmegaConf.load(args.config)
    
    # Load local config overrides if exists (for machine-specific paths)
    local_config_path = args.config.replace('.yaml', '.local.yaml')
    if os.path.exists(local_config_path):
        if rank == 0:
            logger.info(f"Loading local config overrides from {local_config_path}")
        local_cfg = OmegaConf.load(local_config_path)
        cfg = OmegaConf.merge(cfg, local_cfg)
    else:
        if rank == 0:
            logger.warning(f"No local config found at {local_config_path}")
            logger.warning(f"Create {local_config_path} with machine-specific paths")
    
    # Apply command-line overrides
    if args.override:
        for override in args.override:
            if '=' in override:
                key, value = override.split('=', 1)
                # Handle lowercase boolean values (true/false) before eval
                if value.lower() == 'true':
                    value = 'True'
                elif value.lower() == 'false':
                    value = 'False'
                value = eval(value)  # Handles int, float, bool, lists, etc.
                OmegaConf.update(cfg, key, value, merge=True)
            else:
                if rank == 0:
                    logger.warning(f"Ignoring invalid override: {override}")
    
    # Convert to dict for compatibility with existing code
    cfg = OmegaConf.to_container(cfg, resolve=True)
    
    # Special handling for distributed training
    if world_size == 1 and 'hardware.device' in ' '.join(args.override):
        # Device already overridden, skip
        pass
    
    # Set random seed
    setup_seed(cfg['seed'])
    
    # Override wandb setting
    if args.no_wandb:
        cfg['logging']['use_wandb'] = False
    
    # Create trainer with distributed parameters
    trainer = TrajectoryDiffusionTrainer(cfg, rank=rank, world_size=world_size, local_rank=local_rank)
    
    # Resume from checkpoint if provided
    if args.resume:
        trainer.load_checkpoint(args.resume)
        all_actions_min, all_actions_max = trainer.action_min, trainer.action_max
    else:
        # get statistics for action
        all_actions_min, all_actions_max = trainer.get_action_statistics()

        # Store action statistics in trainer for checkpointing
        trainer.action_min = all_actions_min
        trainer.action_max = all_actions_max

    # set min and max of diffusion policy normalization
    if hasattr(trainer.model, "module"):
        trainer.model.module.diffusion_decoder.set_data_act_statistics(all_actions_max, all_actions_min)
    else:
        trainer.model.diffusion_decoder.set_data_act_statistics(all_actions_max, all_actions_min)

    # Start training
    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        trainer.save_checkpoint(trainer.start_epoch, is_final=True)
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise
    finally:
        # Cleanup
        if wandb.run:
            wandb.finish()
        cleanup_distributed()  # ADD


if __name__ == "__main__":
    main()
