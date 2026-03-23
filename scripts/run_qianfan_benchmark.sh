#!/bin/bash

# Runs qianfan benchmark, measuring both olmOCR-bench performance and per document processing performance
# Uses baidu/Qianfan-OCR model served via vllm with OpenAI-compatible API
# Reference: https://huggingface.co/baidu/Qianfan-OCR
#            https://github.com/baidubce/skills/blob/develop/skills/qianfanocr-document-intelligence/scripts/qianfan_ocr_cli.py
#
# Usage:
#   ./scripts/run_qianfan_benchmark.sh                                      # Use default model
#   ./scripts/run_qianfan_benchmark.sh --benchrepo allenai/olmOCR-bench-internal  # Use different benchmark repo
#   ./scripts/run_qianfan_benchmark.sh --benchbranch olmOCR-bench-1125      # Use specific branch/revision
#   ./scripts/run_qianfan_benchmark.sh --benchpath s3://ai2-oe-data/path/   # Use benchmark from S3 or local path
#   ./scripts/run_qianfan_benchmark.sh --cluster ai2/titan-cirrascale       # Specify a cluster
#   ./scripts/run_qianfan_benchmark.sh --beaker-image jakep/olmocr-benchmark-0.3.3-780bc7d934  # Skip Docker build
#   ./scripts/run_qianfan_benchmark.sh --noperf                             # Skip the performance test job
#   ./scripts/run_qianfan_benchmark.sh --model baidu/Qianfan-OCR            # Use a specific model

set -e

# Parse command line arguments
BENCH_BRANCH=""
BENCH_REPO=""
BENCH_PATH=""
CLUSTER=""
BEAKER_IMAGE=""
NOPERF=""
MODEL=""

while [[ $# -gt 0 ]]; do
    case $1 in
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
        --cluster)
            CLUSTER="$2"
            shift 2
            ;;
        --beaker-image)
            BEAKER_IMAGE="$2"
            shift 2
            ;;
        --noperf)
            NOPERF="1"
            shift
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--benchbranch BRANCH] [--benchrepo REPO] [--benchpath PATH] [--cluster CLUSTER] [--beaker-image IMAGE] [--noperf] [--model MODEL]"
            exit 1
            ;;
    esac
done

# Check for mutual exclusivity between benchpath and benchrepo/benchbranch
if [ -n "$BENCH_PATH" ] && ([ -n "$BENCH_REPO" ] || [ -n "$BENCH_BRANCH" ]); then
    echo "Error: --benchpath is mutually exclusive with --benchrepo and --benchbranch"
    echo "Use either --benchpath OR --benchrepo/--benchbranch, not both."
    exit 1
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
    IMAGE_TAG="olmocr-benchmark-${VERSION}-${GIT_HASH}"
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
cat << 'EOF' > /tmp/run_qianfan_benchmark_experiment.py
import sys
from textwrap import dedent
from beaker import Beaker, BeakerExperimentSpec, BeakerTaskSpec, BeakerTaskContext, BeakerResultSpec, BeakerTaskResources, BeakerImageSource, BeakerJobPriority, BeakerConstraints, BeakerEnvVar

# Get image tag, beaker user, git branch, git hash from command line
image_tag = sys.argv[1]
beaker_user = sys.argv[2]
git_branch = sys.argv[3]
git_hash = sys.argv[4]

# Initialize benchmark dataset parameters
bench_branch = None
bench_repo = "allenai/olmOCR-bench"  # Default repository
bench_path = None
cluster = None
noperf = False
model = None

# Parse additional arguments
arg_idx = 5
while arg_idx < len(sys.argv):
    if sys.argv[arg_idx] == "--benchbranch":
        bench_branch = sys.argv[arg_idx + 1]
        arg_idx += 2
    elif sys.argv[arg_idx] == "--benchrepo":
        bench_repo = sys.argv[arg_idx + 1]
        arg_idx += 2
    elif sys.argv[arg_idx] == "--benchpath":
        bench_path = sys.argv[arg_idx + 1]
        arg_idx += 2
    elif sys.argv[arg_idx] == "--cluster":
        cluster = sys.argv[arg_idx + 1]
        arg_idx += 2
    elif sys.argv[arg_idx] == "--noperf":
        noperf = True
        arg_idx += 1
    elif sys.argv[arg_idx] == "--model":
        model = sys.argv[arg_idx + 1]
        arg_idx += 2
    else:
        print(f"Unknown argument: {sys.argv[arg_idx]}")
        arg_idx += 1

# Default model for Qianfan OCR
qianfan_model = model if model else "baidu/Qianfan-OCR"

# Initialize Beaker client
b = Beaker.from_env(default_workspace="ai2/olmocr")

# Check if AWS credentials secret exists
aws_creds_secret = f"{beaker_user}-AWS_CREDENTIALS_FILE"
try:
    b.secret.get(aws_creds_secret, workspace="ai2/olmocr")
    has_aws_creds = True
    print(f"Found AWS credentials secret: {aws_creds_secret}")
except:
    has_aws_creds = False
    print(f"AWS credentials secret not found: {aws_creds_secret}")

# Check if HF_TOKEN secret exists
hf_token_secret = f"{beaker_user}-HF_TOKEN"
try:
    b.secret.get(hf_token_secret, workspace="ai2/olmocr")
    has_hf_token = True
    print(f"Found HuggingFace token secret: {hf_token_secret}")
except:
    has_hf_token = False
    print(f"HuggingFace token secret not found: {hf_token_secret}")

# Shell script to run Qianfan OCR conversions for benchmark
# Uses scripts/qianfan_bench_convert.py which is baked into the Docker image
run_qianfan_shell = dedent("""\
bash -lc 'set -euo pipefail
PDF_ROOT="olmOCR-bench/bench_data/pdfs"
TARGET_ROOT="olmOCR-bench/bench_data/qianfan"
rm -rf "$TARGET_ROOT"
mkdir -p "$TARGET_ROOT"

# Start vllm server in background
echo "Starting vllm server for Qianfan OCR..."
vllm serve __QIANFAN_MODEL__ --trust-remote-code --served-model-name qianfan-ocr --max-model-len 16384 > /tmp/vllm_server.log 2>&1 &
VLLM_PID=$!

# Wait for vllm server to be ready
echo "Waiting for vllm server to start..."
for i in {1..600}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "vllm server is ready"
        break
    fi
    if [ $i -eq 600 ]; then
        echo "Error: vllm server failed to start after 600 seconds"
        cat /tmp/vllm_server.log
        exit 1
    fi
    sleep 1
done

python scripts/qianfan_bench_convert.py "$PDF_ROOT" "$TARGET_ROOT"

# Kill vllm server
echo "Stopping vllm server..."
kill $VLLM_PID || true
wait $VLLM_PID 2>/dev/null || true
'""").replace("__QIANFAN_MODEL__", qianfan_model)

# First experiment: Original benchmark job
commands = []
if has_aws_creds:
    commands.extend([
        "mkdir -p ~/.aws",
        'echo "$AWS_CREDENTIALS_FILE" > ~/.aws/credentials'
    ])

if has_hf_token:
    commands.append('export HF_TOKEN="$HF_TOKEN"')

# Install uv for fast dependency management, then s5cmd (needed for S3 operations)
commands.append("pip install uv")
commands.append("uv pip install s5cmd")

# Handle benchmark data download based on source type
if bench_path:
    if bench_path.startswith("s3://"):
        commands.append(f"s5cmd cp {bench_path.rstrip('/')}/* ./olmOCR-bench/")
    else:
        commands.append(f"cp -r {bench_path} ./olmOCR-bench")
else:
    hf_download_cmd = f"hf download --repo-type dataset {bench_repo} --max-workers 2"
    if bench_branch:
        hf_download_cmd += f" --revision {bench_branch}"
    hf_download_cmd += " --local-dir ./olmOCR-bench"
    commands.append(hf_download_cmd)

# Install poppler-utils for pdftoppm (PDF to image conversion), upgrade vllm, then run
commands.extend([
    "apt-get update && apt-get install -y poppler-utils",
    "uv pip install --upgrade vllm",
    run_qianfan_shell,
    "python -m olmocr.bench.benchmark --dir ./olmOCR-bench/bench_data --candidate qianfan"
])

# Build task spec with optional env vars
# If image_tag contains '/', it's already a full beaker image reference
if '/' in image_tag:
    image_ref = image_tag
else:
    image_ref = f"{beaker_user}/{image_tag}"

task_spec_args = {
    "name": "qianfan-benchmark",
    "image": BeakerImageSource(beaker=image_ref),
    "command": [
        "bash", "-c",
        " && ".join(commands)
    ],
    "context": BeakerTaskContext(
        priority=BeakerJobPriority["normal"],
        preemptible=True,
    ),
    "resources": BeakerTaskResources(gpu_count=1),
    "constraints": BeakerConstraints(cluster=[cluster] if cluster else ["ai2/ceres-cirrascale", "ai2/jupiter-cirrascale-2"]),
    "result": BeakerResultSpec(path="/noop-results"),
}

# Add env vars if AWS credentials or HF token exist
env_vars = []
if has_aws_creds:
    env_vars.append(BeakerEnvVar(name="AWS_CREDENTIALS_FILE", secret=aws_creds_secret))
if has_hf_token:
    env_vars.append(BeakerEnvVar(name="HF_TOKEN", secret=hf_token_secret))
if env_vars:
    task_spec_args["env_vars"] = env_vars

# Create first experiment spec
experiment_spec = BeakerExperimentSpec(
    description=f"Qianfan OCR Benchmark Run - Branch: {git_branch}, Commit: {git_hash}, Model: {qianfan_model}",
    budget="ai2/oe-base",
    tasks=[BeakerTaskSpec(**task_spec_args)],
)

# Create the first experiment
workload = b.experiment.create(spec=experiment_spec, workspace="ai2/olmocr")
print(f"Created benchmark experiment: {workload.experiment.id}")
print(f"View at: https://beaker.org/ex/{workload.experiment.id}")
print("-------")
print("")

# Second experiment: Performance test job (only if --noperf not specified)
if not noperf:
    perf_commands = []
    if has_aws_creds:
        perf_commands.extend([
            "mkdir -p ~/.aws",
            'echo "$AWS_CREDENTIALS_FILE" > ~/.aws/credentials'
        ])

    if has_hf_token:
        perf_commands.append('export HF_TOKEN="$HF_TOKEN"')

    # Shell script for performance test
    perf_shell = dedent("""\
set -euo pipefail

# Start vllm server in background
echo "Starting vllm server for Qianfan OCR..."
vllm serve __QIANFAN_MODEL__ --trust-remote-code --served-model-name qianfan-ocr --max-model-len 16384 > /tmp/vllm_server.log 2>&1 &
VLLM_PID=$!

# Wait for vllm server to be ready
echo "Waiting for vllm server to start..."
for i in {1..600}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "vllm server is ready"
        break
    fi
    if [ $i -eq 600 ]; then
        echo "Error: vllm server failed to start after 600 seconds"
        cat /tmp/vllm_server.log
        exit 1
    fi
    sleep 1
done

# Run the performance test
time python scripts/qianfan_bench_convert.py /root/olmOCR-mix-0225_benchmark_set/ /root/olmOCR-mix-0225_benchmark_set_qianfan

# Kill vllm server
echo "Stopping vllm server..."
kill $VLLM_PID || true
wait $VLLM_PID 2>/dev/null || true
""").replace("__QIANFAN_MODEL__", qianfan_model)

    perf_commands.extend([
        "pip install uv",
        "apt-get update && apt-get install -y poppler-utils",
        "uv pip install --upgrade vllm",
        "uv pip install awscli",
        "aws s3 cp --recursive s3://ai2-oe-data/jakep/olmocr/olmOCR-mix-0225/benchmark_set/ /root/olmOCR-mix-0225_benchmark_set/",
        f"bash -c '{perf_shell}'"
    ])

    # Build performance task spec
    perf_task_spec_args = {
        "name": "qianfan-performance",
        "image": BeakerImageSource(beaker=image_ref),
        "command": [
            "bash", "-c",
            " && ".join(perf_commands)
        ],
        "context": BeakerTaskContext(
            priority=BeakerJobPriority["normal"],
            preemptible=True,
        ),
        # Need to reserve all 8 gpus for performance spec or else benchmark results can be off (1 for titan-cirrascale)
        "resources": BeakerTaskResources(gpu_count=1 if cluster == "ai2/titan-cirrascale" else 8),
        "constraints": BeakerConstraints(cluster=[cluster] if cluster else ["ai2/ceres-cirrascale", "ai2/jupiter-cirrascale-2"]),
        "result": BeakerResultSpec(path="/noop-results"),
    }

    # Add env vars if AWS credentials or HF token exist
    env_vars = []
    if has_aws_creds:
        env_vars.append(BeakerEnvVar(name="AWS_CREDENTIALS_FILE", secret=aws_creds_secret))
    if has_hf_token:
        env_vars.append(BeakerEnvVar(name="HF_TOKEN", secret=hf_token_secret))
    if env_vars:
        perf_task_spec_args["env_vars"] = env_vars

    # Create performance experiment spec
    perf_experiment_spec = BeakerExperimentSpec(
        description=f"Qianfan OCR Performance Test - Branch: {git_branch}, Commit: {git_hash}, Model: {qianfan_model}",
        budget="ai2/oe-base",
        tasks=[BeakerTaskSpec(**perf_task_spec_args)],
    )

    # Create the performance experiment
    perf_workload = b.experiment.create(spec=perf_experiment_spec, workspace="ai2/olmocr")
    print(f"Created performance experiment: {perf_workload.experiment.id}")
    print(f"View at: https://beaker.org/ex/{perf_workload.experiment.id}")
else:
    print("Skipping performance test (--noperf flag specified)")
EOF

# Run the Python script to create the experiments
echo "Creating Beaker experiments..."

# Build command with appropriate arguments
CMD="$PYTHON /tmp/run_qianfan_benchmark_experiment.py $IMAGE_TAG $BEAKER_USER $GIT_BRANCH $GIT_HASH"

if [ -n "$BENCH_BRANCH" ]; then
    echo "Using bench branch: $BENCH_BRANCH"
    CMD="$CMD --benchbranch \"$BENCH_BRANCH\""
fi

if [ -n "$BENCH_REPO" ]; then
    echo "Using bench repo: $BENCH_REPO"
    CMD="$CMD --benchrepo \"$BENCH_REPO\""
fi

if [ -n "$BENCH_PATH" ]; then
    echo "Using bench path: $BENCH_PATH"
    CMD="$CMD --benchpath \"$BENCH_PATH\""
fi

if [ -n "$CLUSTER" ]; then
    echo "Using cluster: $CLUSTER"
    CMD="$CMD --cluster $CLUSTER"
fi

if [ -n "$NOPERF" ]; then
    echo "Skipping performance tests"
    CMD="$CMD --noperf"
fi

if [ -n "$MODEL" ]; then
    echo "Using model: $MODEL"
    CMD="$CMD --model $MODEL"
fi

eval $CMD

# Clean up temporary file
rm /tmp/run_qianfan_benchmark_experiment.py

echo "Benchmark experiments submitted successfully!"
