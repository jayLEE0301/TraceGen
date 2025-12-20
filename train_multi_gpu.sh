#!/bin/bash

# Multi-GPU training launcher script for trajectory flow training
# Usage: ./train_multi_gpu.sh <config_file> [num_gpus] [additional_args]

set -e

# Default values
CONFIG_FILE=""
NUM_GPUS=2
GPU_IDS=""
ADDITIONAL_ARGS=""

# Parse arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <config_file> [num_gpus] [gpu_ids] [additional_args]"
    echo "Example: $0 cfg/train_system2_flow_cuda2_more_size_data.yaml 4"
    echo "Example: $0 cfg/train_system2_flow_cuda2_more_size_data.yaml 2 \"2,3\""
    echo "Example: $0 cfg/train_system2_flow_cuda2_more_size_data.yaml 1 \"2\""
    exit 1
fi

CONFIG_FILE=$1
NUM_GPUS=${2:-2}

# Check if third argument looks like GPU IDs (contains digits and commas)
if [ $# -ge 3 ] && [[ "$3" =~ ^[0-9,]+$ ]]; then
    GPU_IDS="$3"
    ADDITIONAL_ARGS="${@:4}"
else
    ADDITIONAL_ARGS="${@:3}"
fi

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file '$CONFIG_FILE' not found!"
    exit 1
fi

# Check if we have enough GPUs
AVAILABLE_GPUS=$(nvidia-smi -L | wc -l)
if [ $NUM_GPUS -gt $AVAILABLE_GPUS ]; then
    echo "Error: Requested $NUM_GPUS GPUs but only $AVAILABLE_GPUS available!"
    exit 1
fi

echo "=================================================="
echo "Multi-GPU Training Configuration"
echo "=================================================="
echo "Config file: $CONFIG_FILE"
echo "Number of GPUs: $NUM_GPUS"
if [ -n "$GPU_IDS" ]; then
    echo "GPU IDs: $GPU_IDS"
else
    echo "GPU IDs: 0-$((NUM_GPUS-1)) (default)"
fi
echo "Available GPUs: $AVAILABLE_GPUS"
echo "Additional args: $ADDITIONAL_ARGS"
echo "=================================================="

# Set CUDA devices
if [ -n "$GPU_IDS" ]; then
    # Validate GPU IDs
    IFS=',' read -ra GPU_ARRAY <<< "$GPU_IDS"
    if [ ${#GPU_ARRAY[@]} -ne $NUM_GPUS ]; then
        echo "Error: Number of GPU IDs (${#GPU_ARRAY[@]}) doesn't match num_gpus ($NUM_GPUS)!"
        exit 1
    fi

    for gpu_id in "${GPU_ARRAY[@]}"; do
        if [ $gpu_id -ge $AVAILABLE_GPUS ]; then
            echo "Error: GPU ID $gpu_id not available (only $AVAILABLE_GPUS GPUs found)!"
            exit 1
        fi
    done

    export CUDA_VISIBLE_DEVICES="$GPU_IDS"
else
    # Use first NUM_GPUS GPUs
    export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
fi

# Single GPU training (fallback)
if [ $NUM_GPUS -eq 1 ]; then
    echo "Starting single GPU training..."
    python train_system2_flow_multigpu.py --config "$CONFIG_FILE" $ADDITIONAL_ARGS
else
    # Multi-GPU training with torchrun
    echo "Starting multi-GPU training with $NUM_GPUS GPUs..."

    # Use torchrun for distributed training
    torchrun \
        --standalone \
        --nproc_per_node=$NUM_GPUS \
        train_system2_flow_multigpu.py \
        --config "$CONFIG_FILE" \
        $ADDITIONAL_ARGS
fi

echo "Training completed!"
