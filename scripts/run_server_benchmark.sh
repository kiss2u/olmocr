#!/bin/bash

# Runs an olmocr-bench server benchmark run with vLLM model serving
#  Basic usage with default model:
#   ./scripts/run_server_benchmark.sh
#  With custom vLLM model and served name:
#   ./scripts/run_server_benchmark.sh --model facebook/opt-125m --served-model-name opt-125m
#  With custom benchmark dataset:
#   ./scripts/run_server_benchmark.sh --benchrepo allenai/olmOCR-bench-internal --model gpt2
#   ./scripts/run_server_benchmark.sh --benchbranch olmOCR-bench-1125 --model gpt2
#   ./scripts/run_server_benchmark.sh --benchpath s3://ai2-oe-data/jakep/olmocr/olmOCR-bench-1125/ --model gpt2
#  With beaker secrets for API keys (format: ENV_VAR=secret-name):
#   ./scripts/run_server_benchmark.sh --beaker-secret OPENAI_API_KEY=jakep-openai-key --model gpt2
#  With cluster parameter: specify a specific cluster to use
#   ./scripts/run_server_benchmark.sh --cluster ai2/titan-cirrascale --model gpt2
#  With beaker image: skip Docker build and use provided Beaker image
#   ./scripts/run_server_benchmark.sh --beaker-image jakep/olmocr-benchmark-0.3.3-780bc7d934 --model gpt2
#  With additional server convert arguments:
#   ./scripts/run_server_benchmark.sh --model gpt2 server:name=test1

set -e

# Parse command line arguments
CLUSTER=""
BENCH_BRANCH=""
BENCH_REPO=""
BENCH_PATH=""
BEAKER_IMAGE=""
VLLM_MODEL=""
SERVED_MODEL_NAME=""
BEAKER_SECRETS=()
CONVERT_ARGS=()

# First pass: extract our known arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cluster)
            CLUSTER="$2"
            shift 2
            ;;
        --benchbranch)
            BENCH_BRANCH="$2"
            shift 2
            ;;
        --benchrepo)
            BENCH_REPO="$2"
            shift 2
            ;;
        --benchpath)
            BENCH_PATH="$2"
            shift 2
            ;;
        --beaker-image)
            BEAKER_IMAGE="$2"
            shift 2
            ;;
        --model)
            VLLM_MODEL="$2"
            shift 2
            ;;
        --served-model-name)
            SERVED_MODEL_NAME="$2"
            shift 2
            ;;
        --beaker-secret)
            # Format: ENV_VAR=secret-name
            BEAKER_SECRETS+=("$2")
            shift 2
            ;;
        *)
            # Store args to forward to convert
            CONVERT_ARGS+=("$1")
            shift
            ;;
    esac
done

# Set default values if not provided
if [ -z "$VLLM_MODEL" ]; then
    echo "Error: --model argument is required to specify the vLLM model to serve"
    echo ""
    echo "Usage examples:"
    echo "  ./scripts/run_server_benchmark.sh --model facebook/opt-125m"
    echo "  ./scripts/run_server_benchmark.sh --model gpt2 --served-model-name my-gpt2"
    echo "  ./scripts/run_server_benchmark.sh --model meta-llama/Llama-2-7b-hf server:name=llama2"
    exit 1
fi

# If served-model-name not specified, use the model name
if [ -z "$SERVED_MODEL_NAME" ]; then
    SERVED_MODEL_NAME="$VLLM_MODEL"
    echo "Using served-model-name: $SERVED_MODEL_NAME"
fi

# Check for mutual exclusivity between benchpath and benchrepo/benchbranch
if [ -n "$BENCH_PATH" ] && ([ -n "$BENCH_REPO" ] || [ -n "$BENCH_BRANCH" ]); then
    echo "Error: --benchpath is mutually exclusive with --benchrepo and --benchbranch"
    echo "Use either --benchpath OR --benchrepo/--benchbranch, not both."
    exit 1
fi

# Check for uncommitted changes
if [ -n "$BEAKER_IMAGE" ]; then
    echo "Skipping docker build"
else
    if ! git diff-index --quiet HEAD --; then
        echo "Error: There are uncommitted changes in the repository."
        echo "Please commit or stash your changes before running the benchmark."
        echo ""
        echo "Uncommitted changes:"
        git status --short
        exit 1
    fi
fi

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

# Check if a Beaker image was provided
if [ -n "$BEAKER_IMAGE" ]; then
    echo "Using provided Beaker image: $BEAKER_IMAGE"
    IMAGE_TAG="$BEAKER_IMAGE"
else
    # Create full image tag
    IMAGE_TAG="olmocr-server-benchmark-${VERSION}-${GIT_HASH}"
    echo "Building Docker image with tag: $IMAGE_TAG"

    # Build the Docker image
    echo "Building Docker image..."
    docker build --platform linux/amd64 -f ./Dockerfile -t $IMAGE_TAG .

    # Push image to beaker
    echo "Trying to push image to Beaker..."
    if ! beaker image create --workspace ai2/oe-data-pdf --name $IMAGE_TAG $IMAGE_TAG 2>/dev/null; then
        echo "Warning: Beaker image with tag $IMAGE_TAG already exists. Using existing image."
    fi
fi

# Get Beaker username
BEAKER_USER=$(beaker account whoami --format json | jq -r '.[0].name')
echo "Beaker user: $BEAKER_USER"

# Create Python script to run beaker experiment
cat << 'EOF' > /tmp/run_server_benchmark_experiment.py
import sys
from textwrap import dedent
from beaker import Beaker, ExperimentSpec, TaskSpec, TaskContext, ResultSpec, TaskResources, ImageSource, Priority, Constraints, EnvVar

# Get image tag, beaker user, git branch, git hash from command line
image_tag = sys.argv[1]
beaker_user = sys.argv[2]
git_branch = sys.argv[3]
git_hash = sys.argv[4]
vllm_model = sys.argv[5]
served_model_name = sys.argv[6]
cluster = None
bench_branch = None
bench_repo = "allenai/olmOCR-bench"  # Default repository
bench_path = None
convert_args = []
beaker_secrets = {}  # Dict of ENV_VAR: secret_name

# Parse remaining arguments
arg_idx = 7
while arg_idx < len(sys.argv):
    if sys.argv[arg_idx] == "--cluster":
        cluster = sys.argv[arg_idx + 1]
        arg_idx += 2
    elif sys.argv[arg_idx] == "--benchbranch":
        bench_branch = sys.argv[arg_idx + 1]
        arg_idx += 2
    elif sys.argv[arg_idx] == "--benchrepo":
        bench_repo = sys.argv[arg_idx + 1]
        arg_idx += 2
    elif sys.argv[arg_idx] == "--benchpath":
        bench_path = sys.argv[arg_idx + 1]
        arg_idx += 2
    elif sys.argv[arg_idx] == "--beaker-secret":
        # Parse ENV_VAR=secret-name format
        secret_spec = sys.argv[arg_idx + 1]
        if "=" in secret_spec:
            env_var, secret_name = secret_spec.split("=", 1)
            beaker_secrets[env_var] = secret_name
        arg_idx += 2
    else:
        # Everything else is a convert arg
        convert_args.append(sys.argv[arg_idx])
        arg_idx += 1

# Initialize Beaker client
b = Beaker.from_env(default_workspace="ai2/olmocr")

# Check if AWS credentials secret exists
aws_creds_secret = f"{beaker_user}-AWS_CREDENTIALS_FILE"
try:
    # Try to get the secret to see if it exists
    b.secret.get(aws_creds_secret, workspace="ai2/olmocr")
    has_aws_creds = True
    print(f"Found AWS credentials secret: {aws_creds_secret}")
except:
    has_aws_creds = False
    print(f"AWS credentials secret not found: {aws_creds_secret}")

# Check if HF_TOKEN secret exists
hf_token_secret = f"{beaker_user}-HF_TOKEN"
try:
    # Try to get the secret to see if it exists
    b.secret.get(hf_token_secret, workspace="ai2/olmocr")
    has_hf_token = True
    print(f"Found HuggingFace token secret: {hf_token_secret}")
except:
    has_hf_token = False
    print(f"HuggingFace token secret not found: {hf_token_secret}")

# Shell script to run server benchmark with vLLM
run_server_shell = dedent(f"""\
bash -lc 'set -euo pipefail

# Start vllm server in background
echo "Starting vllm server for model: {vllm_model}..."
vllm serve {vllm_model} --served-model-name {served_model_name} > /tmp/vllm_server.log 2>&1 &
VLLM_PID=$!

# Wait for vllm server to be ready
echo "Waiting for vllm server to start..."
for i in {{1..600}}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "vllm server is ready"
        break
    fi
    if [ $i -eq 600 ]; then
        echo "Error: vllm server failed to start after 600 seconds"
        echo "Last 100 lines of server log:"
        tail -100 /tmp/vllm_server.log
        exit 1
    fi
    sleep 1
done

# Show server info
echo "vLLM server started successfully"
curl -s http://localhost:8000/v1/models | python -m json.tool || true

# Run the convert command
echo "Running convert with server endpoint..."
python -m olmocr.bench.convert server:model={served_model_name} {"" + " ".join(convert_args) if convert_args else ""} --dir ./olmOCR-bench/bench_data

# Kill vllm server
echo "Stopping vllm server..."
kill $VLLM_PID || true
wait $VLLM_PID 2>/dev/null || true
'""")

# Build commands
commands = []
if has_aws_creds:
    commands.extend([
        "mkdir -p ~/.aws",
        'echo "$AWS_CREDENTIALS_FILE" > ~/.aws/credentials'
    ])

if has_hf_token:
    commands.append('export HF_TOKEN="$HF_TOKEN"')

# Export any beaker secrets as environment variables
for env_var in beaker_secrets:
    commands.append(f'export {env_var}="${env_var}"')

# Install dependencies
commands.extend([
    "pip install s5cmd",
    "pip install --upgrade vllm"  # Ensure vllm is installed
])

# Handle benchmark data download based on source type
if bench_path:
    # If bench_path is provided, use it (can be S3 or local path)
    if bench_path.startswith("s3://"):
        # S3 path - use s5cmd to download
        commands.append(f"s5cmd cp {bench_path.rstrip('/')}/* ./olmOCR-bench/")
    else:
        # Local path - copy directly
        commands.append(f"cp -r {bench_path} ./olmOCR-bench")
else:
    # Use HuggingFace download (default behavior)
    hf_download_cmd = f"hf download --repo-type dataset {bench_repo} --max-workers 2"
    if bench_branch:
        hf_download_cmd += f" --revision {bench_branch}"
    hf_download_cmd += " --local-dir ./olmOCR-bench"
    commands.append(hf_download_cmd)

# Run the server and convert
commands.append(run_server_shell)

# Copy workspace to S3 for archival (using BEAKER_WORKLOAD_ID for unique path)
commands.append("s5cmd cp ./olmOCR-bench/ s3://ai2-oe-data/jakep/olmocr-bench-runs/$BEAKER_WORKLOAD_ID/olmOCR-bench/")

# Run benchmark
commands.append("python -m olmocr.bench.benchmark --dir ./olmOCR-bench/bench_data")

# Build task spec with optional env vars
# If image_tag contains '/', it's already a full beaker image reference
if '/' in image_tag:
    image_ref = image_tag
else:
    image_ref = f"{beaker_user}/{image_tag}"

task_spec_args = {
    "name": "olmocr-server-benchmark",
    "image": ImageSource(beaker=image_ref),
    "command": [
        "bash", "-c",
        " && ".join(commands)
    ],
    "context": TaskContext(
        priority=Priority.normal,
        preemptible=True,
    ),
    "resources": TaskResources(gpu_count=1),  # Need GPU for vLLM
    "constraints": Constraints(cluster=[cluster] if cluster else ["ai2/ceres-cirrascale", "ai2/jupiter-cirrascale-2"]),
    "result": ResultSpec(path="/noop-results"),
}

# Add env vars if AWS credentials or HF token exist
env_vars = []
if has_aws_creds:
    env_vars.append(EnvVar(name="AWS_CREDENTIALS_FILE", secret=aws_creds_secret))
if has_hf_token:
    env_vars.append(EnvVar(name="HF_TOKEN", secret=hf_token_secret))
# Add any additional beaker secrets
for env_var, secret_name in beaker_secrets.items():
    env_vars.append(EnvVar(name=env_var, secret=secret_name))
if env_vars:
    task_spec_args["env_vars"] = env_vars

# Create a readable experiment name
experiment_name = f"server-bench-{vllm_model.replace('/', '-')}"
if len(experiment_name) > 50:
    # Truncate long model names
    experiment_name = f"server-bench-{vllm_model.split('/')[-1]}"

print(f"Experiment name: {experiment_name}")

# Create experiment spec
experiment_spec = ExperimentSpec(
    description=f"OlmOCR Server Benchmark - Model: {vllm_model}, Branch: {git_branch}, Commit: {git_hash}",
    budget="ai2/oe-base",
    tasks=[TaskSpec(**task_spec_args)],
    name=experiment_name,
)

# Create the experiment
experiment = b.experiment.create(spec=experiment_spec, workspace="ai2/olmocr")
print(f"Created server benchmark experiment: {experiment_name} ({experiment.id})")
print(f"View at: https://beaker.org/ex/{experiment.id}")
EOF

# Run the Python script to create the experiment
echo "Creating Beaker experiment..."

# Build command with appropriate arguments
CMD="$PYTHON /tmp/run_server_benchmark_experiment.py '$IMAGE_TAG' '$BEAKER_USER' '$GIT_BRANCH' '$GIT_HASH' '$VLLM_MODEL' '$SERVED_MODEL_NAME'"

if [ -n "$CLUSTER" ]; then
    echo "Using cluster: $CLUSTER"
    CMD="$CMD --cluster '$CLUSTER'"
fi

if [ -n "$BENCH_BRANCH" ]; then
    echo "Using bench branch: $BENCH_BRANCH"
    CMD="$CMD --benchbranch '$BENCH_BRANCH'"
fi

if [ -n "$BENCH_REPO" ]; then
    echo "Using bench repo: $BENCH_REPO"
    CMD="$CMD --benchrepo '$BENCH_REPO'"
fi

if [ -n "$BENCH_PATH" ]; then
    echo "Using bench path: $BENCH_PATH"
    CMD="$CMD --benchpath '$BENCH_PATH'"
fi

# Add beaker secrets if any
if [ ${#BEAKER_SECRETS[@]} -gt 0 ]; then
    echo "Using beaker secrets:"
    for secret in "${BEAKER_SECRETS[@]}"; do
        echo "  $secret"
        CMD="$CMD --beaker-secret '$secret'"
    done
fi

# Add convert args if any
if [ ${#CONVERT_ARGS[@]} -gt 0 ]; then
    echo "Forwarding to convert: ${CONVERT_ARGS[*]}"
    for arg in "${CONVERT_ARGS[@]}"; do
        CMD="$CMD '$arg'"
    done
fi

eval $CMD

# Clean up temporary file
rm /tmp/run_server_benchmark_experiment.py

echo "Server Benchmark experiment submitted successfully!"