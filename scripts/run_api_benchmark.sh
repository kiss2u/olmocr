#!/bin/bash

# Runs an olmocr-bench API benchmark run
#  Without additional parameters (default behavior): uses the default benchmark dataset from hugging face
#   ./scripts/run_api_benchmark.sh
#  With API provider arguments (forwarded to olmocr.bench.convert):
#   ./scripts/run_api_benchmark.sh chatgpt:name=test1:prompt=long
#   ./scripts/run_api_benchmark.sh gemini:name=test2:api_key=YOUR_KEY
#  With cluster parameter: specify a specific cluster to use
#   ./scripts/run_api_benchmark.sh --cluster ai2/titan-cirrascale chatgpt:name=test1:prompt=long
#  With beaker image: skip Docker build and use provided Beaker image
#   ./scripts/run_api_benchmark.sh --beaker-image jakep/olmocr-benchmark-0.3.3-780bc7d934 chatgpt:name=test1
#  With benchrepo parameter: use a different benchmark dataset repository (default: allenai/olmOCR-bench)
#   ./scripts/run_api_benchmark.sh --benchrepo allenai/olmOCR-bench-internal chatgpt:name=test1
#  With benchbranch parameter: use a specific branch/revision of the benchmark dataset
#   ./scripts/run_api_benchmark.sh --benchbranch olmOCR-bench-1125 chatgpt:name=test1
#  With benchpath parameter: use benchmark dataset from a local path or S3 path (mutually exclusive with benchrepo/benchbranch)
#   ./scripts/run_api_benchmark.sh --benchpath s3://ai2-oe-data/jakep/olmocr/olmOCR-bench-1125/ chatgpt:name=test1
#  All other arguments are forwarded to olmocr.bench.convert

set -e

# Parse command line arguments
CLUSTER=""
BENCH_BRANCH=""
BENCH_REPO=""
BENCH_PATH=""
BEAKER_IMAGE=""
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
        *)
            # Store args to forward to convert
            CONVERT_ARGS+=("$1")
            shift
            ;;
    esac
done

# Check for mutual exclusivity between benchpath and benchrepo/benchbranch
if [ -n "$BENCH_PATH" ] && ([ -n "$BENCH_REPO" ] || [ -n "$BENCH_BRANCH" ]); then
    echo "Error: --benchpath is mutually exclusive with --benchrepo and --benchbranch"
    echo "Use either --benchpath OR --benchrepo/--benchbranch, not both."
    exit 1
fi

# Check that we have at least one convert argument
if [ ${#CONVERT_ARGS[@]} -eq 0 ]; then
    echo "Error: No API provider arguments specified for olmocr.bench.convert"
    echo ""
    echo "Usage examples:"
    echo "  ./scripts/run_api_benchmark.sh chatgpt:name=test1:prompt=long"
    echo "  ./scripts/run_api_benchmark.sh gemini:name=test2:api_key=YOUR_KEY"
    echo "  ./scripts/run_api_benchmark.sh --cluster ai2/titan-cirrascale chatgpt:name=test1"
    echo ""
    echo "You must provide at least one API provider configuration to benchmark."
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
    IMAGE_TAG="olmocr-api-benchmark-${VERSION}-${GIT_HASH}"
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
cat << 'EOF' > /tmp/run_api_benchmark_experiment.py
import sys
from beaker import Beaker, ExperimentSpec, TaskSpec, TaskContext, ResultSpec, TaskResources, ImageSource, Priority, Constraints, EnvVar

# Get image tag, beaker user, git branch, git hash from command line
image_tag = sys.argv[1]
beaker_user = sys.argv[2]
git_branch = sys.argv[3]
git_hash = sys.argv[4]
cluster = None
bench_branch = None
bench_repo = "allenai/olmOCR-bench"  # Default repository
bench_path = None
convert_args = []

# Parse remaining arguments
arg_idx = 5
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
    else:
        # Everything else is a convert arg
        convert_args.append(sys.argv[arg_idx])
        arg_idx += 1

# Validate we have convert args
if not convert_args:
    print("Error: No API provider arguments provided for convert")
    sys.exit(1)

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

# Build commands
commands = []
if has_aws_creds:
    commands.extend([
        "mkdir -p ~/.aws",
        'echo "$AWS_CREDENTIALS_FILE" > ~/.aws/credentials'
    ])

if has_hf_token:
    commands.append('export HF_TOKEN="$HF_TOKEN"')

# Install s5cmd (needed for S3 operations)
commands.append("pip install s5cmd")

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

# Run convert with forwarded args
convert_cmd = "python -m olmocr.bench.convert"
if convert_args:
    convert_cmd += " " + " ".join(convert_args)
convert_cmd += " --dir ./olmOCR-bench/bench_data"
commands.append(convert_cmd)

# Run benchmark
commands.append("python -m olmocr.bench.benchmark --dir ./olmOCR-bench/bench_data")

# Build task spec with optional env vars
# If image_tag contains '/', it's already a full beaker image reference
if '/' in image_tag:
    image_ref = image_tag
else:
    image_ref = f"{beaker_user}/{image_tag}"

task_spec_args = {
    "name": "olmocr-api-benchmark",
    "image": ImageSource(beaker=image_ref),
    "command": [
        "bash", "-c",
        " && ".join(commands)
    ],
    "context": TaskContext(
        priority=Priority.normal,
        preemptible=True,
    ),
    "resources": TaskResources(gpu_count=0),
    "constraints": Constraints(cluster=[cluster] if cluster else ["ai2/phobos"),
    "result": ResultSpec(path="/noop-results"),
}

# Add env vars if AWS credentials or HF token exist
env_vars = []
if has_aws_creds:
    env_vars.append(EnvVar(name="AWS_CREDENTIALS_FILE", secret=aws_creds_secret))
if has_hf_token:
    env_vars.append(EnvVar(name="HF_TOKEN", secret=hf_token_secret))
if env_vars:
    task_spec_args["env_vars"] = env_vars

# Create experiment spec
experiment_spec = ExperimentSpec(
    description=f"OlmOCR API Benchmark Run - Branch: {git_branch}, Commit: {git_hash}",
    budget="ai2/oe-base",
    tasks=[TaskSpec(**task_spec_args)],
)

# Create the experiment
experiment = b.experiment.create(spec=experiment_spec, workspace="ai2/olmocr")
print(f"Created API benchmark experiment: {experiment.id}")
print(f"View at: https://beaker.org/ex/{experiment.id}")
EOF

# Run the Python script to create the experiment
echo "Creating Beaker experiment..."

# Build command with appropriate arguments
CMD="$PYTHON /tmp/run_api_benchmark_experiment.py $IMAGE_TAG $BEAKER_USER $GIT_BRANCH $GIT_HASH"

if [ -n "$CLUSTER" ]; then
    echo "Using cluster: $CLUSTER"
    CMD="$CMD --cluster $CLUSTER"
fi

if [ -n "$BENCH_BRANCH" ]; then
    echo "Using bench branch: $BENCH_BRANCH"
    CMD="$CMD --benchbranch $BENCH_BRANCH"
fi

if [ -n "$BENCH_REPO" ]; then
    echo "Using bench repo: $BENCH_REPO"
    CMD="$CMD --benchrepo $BENCH_REPO"
fi

if [ -n "$BENCH_PATH" ]; then
    echo "Using bench path: $BENCH_PATH"
    CMD="$CMD --benchpath $BENCH_PATH"
fi

# Add convert args if any
if [ ${#CONVERT_ARGS[@]} -gt 0 ]; then
    echo "Forwarding to convert: ${CONVERT_ARGS[*]}"
    for arg in "${CONVERT_ARGS[@]}"; do
        CMD="$CMD \"$arg\""
    done
fi

eval $CMD

# Clean up temporary file
rm /tmp/run_api_benchmark_experiment.py

echo "API Benchmark experiment submitted successfully!"