#!/bin/bash

set -e

# Parse beaker-specific arguments
SKIP_DOCKER_BUILD=false
PREEMPTIBLE=false
EXP_NAME=""
NUM_TRAIN_GPUS=3
NUM_GENERATE_GPUS=1

# Store all arguments to pass to python command
PYTHON_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-docker-build)
            SKIP_DOCKER_BUILD=true
            shift
            ;;
        --preemptible)
            PREEMPTIBLE=true
            shift
            ;;
        --name)
            EXP_NAME="$2"
            shift 2
            ;;
        --num-train-gpus)
            NUM_TRAIN_GPUS="$2"
            if [ "$NUM_TRAIN_GPUS" -lt 1 ] || [ "$NUM_TRAIN_GPUS" -gt 7 ]; then
                echo "Error: --num-train-gpus must be between 1 and 7 (got: $NUM_TRAIN_GPUS)"
                exit 1
            fi
            shift 2
            ;;
        --num-generate-gpus)
            NUM_GENERATE_GPUS="$2"
            if [ "$NUM_GENERATE_GPUS" -lt 1 ] || [ "$NUM_GENERATE_GPUS" -gt 4 ]; then
                echo "Error: --num-generate-gpus must be between 1 and 4 (got: $NUM_GENERATE_GPUS)"
                exit 1
            fi
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [beaker-options] [grpo-training-options]"
            echo ""
            echo "Beaker-specific options:"
            echo "  --skip-docker-build            Skip Docker build"
            echo "  --preemptible                  Use preemptible instances"
            echo "  --name NAME                    Experiment name (used in output directory)"
            echo "  --num-train-gpus N             Number of GPUs for training (1-7, default: 3)"
            echo "  --num-generate-gpus N          Number of GPUs for VLLM generation (1-4, default: 1)"
            echo ""
            echo "All other arguments are forwarded to python -m olmocr.train.grpo_train"
            echo "Run 'python -m olmocr.train.grpo_train --help' to see available training options"
            echo ""
            echo "This Augusta multi-GPU version runs:"
            echo "  - VLLM server with data parallel on N generation GPUs"
            echo "  - Training on M training GPUs with DeepSpeed"
            echo "  - Total GPUs used: M + N"
            echo "  - Outputs saved locally then synced to S3 automatically via --s3_save_path"
            echo ""
            echo "Note: This version is configured for ai2/augusta cluster (no Weka)"
            exit 0
            ;;
        *)
            # Store all other arguments to pass to python command
            PYTHON_ARGS+=("$1")
            shift
            ;;
    esac
done

# Validate total GPU count
TOTAL_GPUS=$((NUM_TRAIN_GPUS + NUM_GENERATE_GPUS))
if [ "$TOTAL_GPUS" -gt 8 ]; then
    echo "Error: Total GPUs (training + generation) cannot exceed 8 (got: $TOTAL_GPUS)"
    echo "  Training GPUs: $NUM_TRAIN_GPUS"
    echo "  Generation GPUs: $NUM_GENERATE_GPUS"
    exit 1
fi

echo "Preemptible: $PREEMPTIBLE"
echo "Skip Docker Build: $SKIP_DOCKER_BUILD"
echo "Number of Training GPUs: $NUM_TRAIN_GPUS"
echo "Number of Generation GPUs: $NUM_GENERATE_GPUS"
echo "Total GPUs: $TOTAL_GPUS"
echo "Arguments to forward: ${PYTHON_ARGS[@]}"

# Use conda environment Python if available, otherwise use system Python
if [ -n "$CONDA_PREFIX" ]; then
    PYTHON="$CONDA_PREFIX/bin/python"
    echo "Using conda Python from: $CONDA_PREFIX"
else
    PYTHON="python"
    echo "Warning: No conda environment detected, using system Python"
fi

# Get version from version.py
VERSION=$($PYTHON -c 'import olmocr.version; print(olmocr.version.VERSION)')
echo "OlmOCR version: $VERSION"

# Get first 10 characters of git hash
GIT_HASH=$(git rev-parse HEAD | cut -c1-10)
echo "Git hash: $GIT_HASH"

# Get current git branch name
GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo "Git branch: $GIT_BRANCH"

# Create full image tag
IMAGE_TAG="olmocr-grpo-${VERSION}-${GIT_HASH}"
echo "Building Docker image with tag: $IMAGE_TAG"

# Build and push Docker image if not skipping
if [ "$SKIP_DOCKER_BUILD" = false ]; then
    echo "Building Docker image..."
    docker build --platform linux/amd64 -f ./Dockerfile -t $IMAGE_TAG .
    
    # Push image to beaker
    echo "Trying to push image to Beaker..."
    if ! beaker image create --workspace ai2/oe-data-pdf --name $IMAGE_TAG $IMAGE_TAG 2>/dev/null; then
        echo "Warning: Beaker image with tag $IMAGE_TAG already exists. Using existing image."
    fi
else
    echo "Skipping Docker build as requested"
fi

# Get Beaker username
BEAKER_USER=$(beaker account whoami --format json | jq -r '.[0].name')
echo "Beaker user: $BEAKER_USER"

# Create Python script to run beaker experiment
cat << 'EOF' > /tmp/run_grpo_experiment_multi_gpu.py
import sys
import shlex
import os
from beaker import Beaker, ExperimentSpec, TaskSpec, TaskContext, ResultSpec, TaskResources, ImageSource, Priority, Constraints, EnvVar

# Get parameters from command line
image_tag = sys.argv[1]
beaker_user = sys.argv[2]
git_branch = sys.argv[3]
git_hash = sys.argv[4]
preemptible = sys.argv[5] == "true"
exp_name = sys.argv[6]  # Empty string if not provided
num_train_gpus = int(sys.argv[7])
num_generate_gpus = int(sys.argv[8])
# All remaining arguments are the python command arguments
python_args = sys.argv[9:]

# Calculate GPU assignments
# Total GPUs needed
num_gpus = num_train_gpus + num_generate_gpus

# Assign first num_train_gpus for training
training_gpus = list(range(num_train_gpus))
training_gpu_str = ",".join(str(g) for g in training_gpus)
num_training_processes = len(training_gpus)

# Assign next num_generate_gpus for VLLM generation
vllm_gpus = list(range(num_train_gpus, num_train_gpus + num_generate_gpus))
vllm_gpu_str = ",".join(str(g) for g in vllm_gpus)

# Initialize Beaker client
b = Beaker.from_env(default_workspace="ai2/olmocr")

# Process arguments to extract model path
model_sync_commands = []
modified_args = list(python_args)
model_path_local = None
for i in range(len(modified_args)):
    if modified_args[i] == "--model_name" and i + 1 < len(modified_args):
        model_path = modified_args[i + 1].rstrip('/')
        if model_path.startswith("s3://"):
            # Extract checkpoint name from S3 path (last part of path)
            checkpoint_name = model_path.split('/')[-1]
            local_model_path = f"/data/models/{checkpoint_name}"
            model_path_local = local_model_path
            
            # Create sync commands
            model_sync_commands = [
                f"echo 'Syncing model from S3: {model_path}'",
                "mkdir -p /data/models",
                f"s5cmd sync '{model_path}/*' '{local_model_path}/'",
            ]
            
            # Replace S3 path with local path in arguments
            modified_args[i + 1] = local_model_path
        else:
            model_path_local = model_path
        break

# Build setup commands
setup_commands = [
    # Install dependencies
    "pip install .[train]",
    "pip install trl wandb",
    "pip install transformers==4.57.1",  # Updated for GRPO compatibility
    "pip install flash-attn --no-build-isolation",
    "pip install vllm==0.10.2",
    "pip install s5cmd",
    "pip install accelerate deepspeed",
    
    # Sync the bench data from S3
    "echo 'Syncing bench data from S3...'",
    "mkdir -p /data/olmOCR-bench",
    "s5cmd sync 's3://ai2-oe-data/jakep/olmocr/olmOCR-bench-snapshot-082225/*' /data/olmOCR-bench/ > /dev/null",
    "s5cmd sync 's3://ai2-oe-data/jakep/grpo_data_mixes/*' /data/jakep/grpo_data_mixes/ > /dev/null",
]

# Add model sync commands if needed
if model_sync_commands:
    setup_commands.extend(model_sync_commands)

# Determine model path for VLLM server
if model_path_local:
    vllm_model_arg = model_path_local
else:
    # Default model if not specified
    vllm_model_arg = "Qwen/Qwen2.5-VL-7B-Instruct"
    for i, arg in enumerate(modified_args):
        if arg == "--model_name" and i + 1 < len(modified_args):
            vllm_model_arg = modified_args[i + 1]
            break

# Extract gradient_accumulation_steps from arguments if provided, otherwise use default
grad_acc_steps = 8  # Default value
for i, arg in enumerate(modified_args):
    if arg == "--gradient_accumulation_steps" and i + 1 < len(modified_args):
        try:
            grad_acc_steps = int(modified_args[i + 1])
        except (ValueError, IndexError):
            pass  # Keep default if parsing fails
        break

# Build the GRPO training command with forwarded arguments
grpo_cmd = f"CUDA_VISIBLE_DEVICES={training_gpu_str} accelerate launch --use_deepspeed --zero_stage 2 --num_processes {num_training_processes} --gradient_accumulation_steps {grad_acc_steps} -m olmocr.train.grpo_train"

# Check if --vllm_mode is specified in arguments
arg_str = " ".join(modified_args)
vllm_mode = "server"  # Default for multi-GPU
for i, arg in enumerate(modified_args):
    if arg == "--vllm_mode" and i + 1 < len(modified_args):
        vllm_mode = modified_args[i + 1]
        break

# Always add --vllm_mode (since we filter it out later)
grpo_cmd += f" --vllm_mode {vllm_mode}"

# Check if certain required arguments are in the provided args, add defaults if not
if "--train_bench_data_folder" not in arg_str:
    grpo_cmd += " --train_bench_data_folder /data/olmOCR-bench/bench_data"
if "--eval_bench_data_folder" not in arg_str:
    grpo_cmd += " --eval_bench_data_folder /data/olmOCR-bench/bench_data"
# Store output folder name for S3 sync
if "--output_dir" not in arg_str:
    # Use local directory for output
    # Note: We'll use the actual BEAKER_WORKLOAD_ID environment variable at runtime
    if exp_name:
        # For multi-GPU runs, add suffix to distinguish
        output_folder_name = f"{exp_name}-multigpu-$BEAKER_WORKLOAD_ID"
    else:
        output_folder_name = f"multigpu-$BEAKER_WORKLOAD_ID"

    # Local output directory (with placeholder for runtime expansion)
    local_output_dir = f"/tmp/checkpoints/{output_folder_name}"
    grpo_cmd += f" --output_dir {local_output_dir}"

    # S3 destination (with placeholder for runtime expansion)
    s3_output_path = f"s3://ai2-oe-data/jakep/olmocr-grpo-checkpoints/{output_folder_name}"
else:
    # Extract output dir from args to determine S3 sync path
    local_output_dir = None
    s3_output_path = None
    for i, arg in enumerate(modified_args):
        if arg == "--output_dir" and i + 1 < len(modified_args):
            local_output_dir = modified_args[i + 1]
            output_folder_name = os.path.basename(local_output_dir)
            s3_output_path = f"s3://ai2-oe-data/jakep/olmocr-grpo-checkpoints/{output_folder_name}"
            break

# Add --s3_save_path parameter if we have an S3 output path
# Check if --s3_save_path is not already in the arguments
if s3_output_path and "--s3_save_path" not in arg_str:
    grpo_cmd += f" --s3_save_path {s3_output_path}"

# Add all the (possibly modified) arguments, filtering out --vllm_mode if it exists to avoid duplicates
# Note: We keep --gradient_accumulation_steps in the args even though we use it for accelerate,
# because the training script also needs it for its configuration
filtered_args = []
skip_next = False
for i, arg in enumerate(modified_args):
    if skip_next:
        skip_next = False
        continue
    if arg == "--vllm_mode":
        skip_next = True  # Skip this and the next argument
        continue
    filtered_args.append(arg)

grpo_cmd += " " + " ".join(filtered_args)

# Create a bash script as a single command string with S3 sync
# S3 sync will be handled directly in the cleanup function

bash_script = f"""
set -e

# Ensure BEAKER_WORKLOAD_ID is available (Beaker sets this automatically)
if [ -z "$BEAKER_WORKLOAD_ID" ]; then
    echo "Warning: BEAKER_WORKLOAD_ID not set, using timestamp as fallback"
    export BEAKER_WORKLOAD_ID=$(date +%Y%m%d-%H%M%S)
fi
echo "BEAKER_WORKLOAD_ID: $BEAKER_WORKLOAD_ID"

# Define cleanup function that will always run
cleanup() {{
    EXIT_CODE=$?
    echo "Running cleanup (exit code: $EXIT_CODE)..."

    # Kill VLLM server if it's still running
    if [ ! -z "$VLLM_PID" ]; then
        echo "Killing VLLM server (PID: $VLLM_PID)..."
        kill $VLLM_PID || true
        echo "VLLM server stopped."
    fi

    # S3 sync is now handled by the training script via --s3_save_path
    echo "Note: S3 sync is handled by the training script's S3SyncCallback"

    if [ $EXIT_CODE -eq 0 ]; then
        echo "Script completed successfully"
    else
        echo "Script failed with exit code: $EXIT_CODE"
    fi

    exit $EXIT_CODE
}}

# Set trap to run cleanup on EXIT (covers both success and failure)
trap cleanup EXIT

# Setup commands
{" && ".join(setup_commands)}

# Create output directory (with runtime variable expansion)
ACTUAL_OUTPUT_DIR="{local_output_dir.replace('$BEAKER_WORKLOAD_ID', '${BEAKER_WORKLOAD_ID}') if local_output_dir else '/tmp/checkpoints'}"
echo "Creating output directory: $ACTUAL_OUTPUT_DIR"
mkdir -p "$ACTUAL_OUTPUT_DIR"

# Sync existing checkpoints from S3 if they exist
S3_PATH="{s3_output_path.replace('$BEAKER_WORKLOAD_ID', '${BEAKER_WORKLOAD_ID}') if s3_output_path else ''}"
RESUME_FLAG=""
if [ ! -z "$S3_PATH" ]; then
    echo "Checking for existing checkpoints in S3: $S3_PATH"
    # Use s5cmd ls to check if the path exists
    if s5cmd ls "$S3_PATH/" 2>/dev/null | grep -q "checkpoint-"; then
        echo "Found existing checkpoints, syncing from S3..."
        s5cmd sync "$S3_PATH/*" "$ACTUAL_OUTPUT_DIR/" || true
        echo "Checkpoint sync complete. Contents of output directory:"
        ls -la "$ACTUAL_OUTPUT_DIR"

        # Check if any checkpoints exist after sync
        if ls -d "$ACTUAL_OUTPUT_DIR"/checkpoint-* 2>/dev/null >/dev/null; then
            echo "Checkpoints found in output directory - will resume training"
            RESUME_FLAG=" --resume_from_checkpoint"
        fi
    else
        echo "No existing checkpoints found in S3"
    fi
fi

# Start VLLM server in background (output goes to console)
echo 'Starting VLLM server on GPUs {vllm_gpu_str} with data parallel...'
CUDA_VISIBLE_DEVICES={vllm_gpu_str} trl vllm-serve --model {vllm_model_arg} --port 8000 --gpu-memory-utilization 0.5 --max-model-len 16384 --data-parallel-size {num_generate_gpus} &
VLLM_PID=$!
echo "VLLM server started with PID: $VLLM_PID"

# Wait for VLLM server to be ready
echo 'Waiting for VLLM server to be ready...'
sleep 30
for i in {{1..60}}; do 
    if curl -s http://localhost:8000/health; then
        echo ' - VLLM server is ready!'
        break
    else
        echo 'Still waiting for VLLM server...'
        sleep 5
    fi
done

# Run training (expand BEAKER_WORKLOAD_ID in the command)
echo 'Starting GRPO training on GPUs {training_gpu_str} ({num_training_processes} processes)...'
echo 'BEAKER_WORKLOAD_ID: '$BEAKER_WORKLOAD_ID
# Replace placeholder with actual workload ID in the command
GRPO_CMD="{grpo_cmd}"
GRPO_CMD="${{GRPO_CMD//\$BEAKER_WORKLOAD_ID/$BEAKER_WORKLOAD_ID}}"
# Add resume flag if checkpoints were found
GRPO_CMD="$GRPO_CMD$RESUME_FLAG"
echo "Running command: $GRPO_CMD"
eval "$GRPO_CMD"

echo 'Training completed successfully!'
"""

# Create single task spec
task_spec = TaskSpec(
    name="olmocr-grpo-multi-gpu",
    image=ImageSource(beaker=f"{beaker_user}/{image_tag}"),
    command=[
        "bash", "-c",
        bash_script
    ],
    context=TaskContext(
        priority=Priority.high,
        preemptible=preemptible,
    ),
    resources=TaskResources(
        gpu_count=num_gpus,  # Request the specified number of GPUs
        shared_memory="10GiB"
    ),
    constraints=Constraints(cluster=["ai2/jupiter", "ai2/augusta"]),
    result=ResultSpec(path="/noop-results"),
    env_vars=[
        EnvVar(name="LOG_FILTER_TYPE", value="local_rank0_only"),
        EnvVar(name="OMP_NUM_THREADS", value="8"),
        EnvVar(name="BEAKER_USER_ID", value=beaker_user),
        EnvVar(name="AWS_ACCESS_KEY_ID", secret="ALLENNLP_AWS_ACCESS_KEY_ID"),
        EnvVar(name="AWS_SECRET_ACCESS_KEY", secret="ALLENNLP_AWS_SECRET_ACCESS_KEY"),
        EnvVar(name="WANDB_API_KEY", secret="JAKE_WANDB_API_KEY"),
    ]
)

# Extract model name from arguments if provided (for description)
model_name = "Unknown"
for i, arg in enumerate(modified_args):
    if arg in ["--model_name", "--model"]:
        if i + 1 < len(modified_args):
            model_name = modified_args[i + 1]
            break

# Create experiment spec with single task
experiment_spec = ExperimentSpec(
    description=f"OlmOCR GRPO Multi-GPU Training ({num_train_gpus} train GPUs + {num_generate_gpus} VLLM GPUs) - Model: {model_name}, Branch: {git_branch}, Commit: {git_hash}",
    budget="ai2/oe-base",
    tasks=[task_spec],  # Single task that manages both VLLM and training
)

# Create the experiment
experiment = b.experiment.create(spec=experiment_spec, workspace="ai2/olmocr")
print(f"Created multi-GPU GRPO training experiment: {experiment.id}")
print(f"View at: https://beaker.org/ex/{experiment.id}")
EOF

# Run the Python script to create the experiment
echo "Creating Beaker multi-GPU GRPO experiment..."
$PYTHON /tmp/run_grpo_experiment_multi_gpu.py \
    "$IMAGE_TAG" \
    "$BEAKER_USER" \
    "$GIT_BRANCH" \
    "$GIT_HASH" \
    "$PREEMPTIBLE" \
    "$EXP_NAME" \
    "$NUM_TRAIN_GPUS" \
    "$NUM_GENERATE_GPUS" \
    "${PYTHON_ARGS[@]}"

# Clean up temporary file
rm /tmp/run_grpo_experiment_multi_gpu.py

echo "Multi-GPU GRPO training experiment submitted successfully!"