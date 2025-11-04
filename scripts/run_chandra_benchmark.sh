#!/bin/bash

# Runs chandra benchmark, measuring both olmOCR-bench performance and per document processing performance
#   ./scripts/run_chandra_benchmark.sh
#   ./scripts/run_chandra_benchmark.sh 0.1.0

set -e

# Parse command line arguments
CHANDRA_VERSION="${1:-latest}"
if [ "$CHANDRA_VERSION" = "latest" ]; then
    echo "Using latest chandra-ocr release"
    CHANDRA_INSTALL_CMD="pip install chandra-ocr"
else
    echo "Using chandra-ocr version: $CHANDRA_VERSION"
    CHANDRA_INSTALL_CMD="pip install chandra-ocr==$CHANDRA_VERSION"
fi

# Check for uncommitted changes
if ! git diff-index --quiet HEAD --; then
    echo "Error: There are uncommitted changes in the repository."
    echo "Please commit or stash your changes before running the benchmark."
    echo ""
    echo "Uncommitted changes:"
    git status --short
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

# Create full image tag
IMAGE_TAG="olmocr-benchmark-${VERSION}-${GIT_HASH}"
echo "Building Docker image with tag: $IMAGE_TAG"

# Build the Docker image
echo "Building Docker image..."
docker build --platform linux/amd64 -f ./Dockerfile -t $IMAGE_TAG .

# Get Beaker username
BEAKER_USER=$(beaker account whoami --format json | jq -r '.[0].name')
echo "Beaker user: $BEAKER_USER"

# Push image to beaker
echo "Trying to push image to Beaker..."
if ! beaker image create --workspace ai2/oe-data-pdf --name $IMAGE_TAG $IMAGE_TAG 2>/dev/null; then
    echo "Warning: Beaker image with tag $IMAGE_TAG already exists. Using existing image."
fi

# Create Python script to run beaker experiment
cat << 'EOF' > /tmp/run_benchmark_experiment.py
import sys
from textwrap import dedent
from beaker import Beaker, ExperimentSpec, TaskSpec, TaskContext, ResultSpec, TaskResources, ImageSource, Priority, Constraints, EnvVar

# Get image tag, beaker user, git branch, git hash, and chandra version from command line
image_tag = sys.argv[1]
beaker_user = sys.argv[2]
git_branch = sys.argv[3]
git_hash = sys.argv[4]
chandra_version = sys.argv[5]
chandra_install_cmd = sys.argv[6]

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

# Shell script to run Chandra conversions for benchmark
run_chandra_shell = dedent("""\
bash -lc 'set -euo pipefail
PDF_ROOT="olmOCR-bench/bench_data/pdfs"
TARGET_ROOT="olmOCR-bench/bench_data/chandra"
rm -rf "$TARGET_ROOT"
mkdir -p "$TARGET_ROOT"

# Start vllm server in background
echo "Starting vllm server for Chandra..."
vllm serve datalab-to/chandra --served-model-name chandra > /tmp/vllm_server.log 2>&1 &
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

# Process each folder separately to maintain structure
echo "Running Chandra conversions..."
for folder in "$PDF_ROOT"/*; do
    if [ ! -d "$folder" ]; then
        continue
    fi
    section=$(basename "$folder")
    output_dir="$HOME/chandra_bench_${section}"
    rm -rf "$output_dir"
    mkdir -p "$output_dir"

    echo "  Processing $folder -> $output_dir"
    chandra "$folder" "$output_dir" --method vllm
done

echo "Collecting Chandra markdown outputs..."
# For each PDF, find its corresponding markdown and copy to proper location
find "$PDF_ROOT" -type f -name "*.pdf" | while IFS= read -r pdf_path; do
    # Get relative path from PDF root
    rel_path=${pdf_path#"$PDF_ROOT"/}

    # Extract section name (first directory level)
    case "$rel_path" in
        */*)
            section=${rel_path%%/*}
            ;;
        *)
            echo "Warning: Unexpected PDF path layout for $pdf_path, skipping" >&2
            continue
            ;;
    esac

    # Get PDF name without extension
    pdf_name=$(basename "$pdf_path" .pdf)

    # Source markdown location: ~/chandra_bench_${section}/${pdf_name}/${pdf_name}.md
    src_md="$HOME/chandra_bench_${section}/${pdf_name}/${pdf_name}.md"

    if [ ! -f "$src_md" ]; then
        echo "Warning: No markdown output found at $src_md" >&2
        continue
    fi

    # Target location: olmOCR-bench/bench_data/chandra/${section}/${pdf_name}_pg1_repeat1.md
    target_dir="$TARGET_ROOT/$section"
    mkdir -p "$target_dir"
    target_path="$target_dir/${pdf_name}_pg1_repeat1.md"

    cp "$src_md" "$target_path"
    echo "  Copied $src_md -> $target_path"
done

# Kill vllm server
echo "Stopping vllm server..."
kill $VLLM_PID || true
wait $VLLM_PID 2>/dev/null || true
'""")

# First experiment: Original benchmark job
commands = []
if has_aws_creds:
    commands.extend([
        "mkdir -p ~/.aws",
        'echo "$AWS_CREDENTIALS_FILE" > ~/.aws/credentials'
    ])
commands.extend([
    "git clone https://huggingface.co/datasets/allenai/olmOCR-bench",
    "cd olmOCR-bench && git lfs pull && cd ..",
    chandra_install_cmd,
    "pip install --upgrade vllm",  # Ensure vllm is installed
    run_chandra_shell,
    "python -m olmocr.bench.benchmark --dir ./olmOCR-bench/bench_data --candidate chandra"
])

# Build task spec with optional env vars
task_spec_args = {
    "name": "chandra-benchmark",
    "image": ImageSource(beaker=f"{beaker_user}/{image_tag}"),
    "command": [
        "bash", "-c",
        " && ".join(commands)
    ],
    "context": TaskContext(
        priority=Priority.normal,
        preemptible=True,
    ),
    "resources": TaskResources(gpu_count=1),
    "constraints": Constraints(cluster=["ai2/ceres-cirrascale", "ai2/jupiter-cirrascale-2"]),
    "result": ResultSpec(path="/noop-results"),
}

# Add env vars if AWS credentials exist
if has_aws_creds:
    task_spec_args["env_vars"] = [
        EnvVar(name="AWS_CREDENTIALS_FILE", secret=aws_creds_secret)
    ]

# Create first experiment spec
chandra_version_label = "latest" if chandra_version == "latest" else chandra_version
experiment_spec = ExperimentSpec(
    description=f"Chandra {chandra_version_label} Benchmark Run - Branch: {git_branch}, Commit: {git_hash}",
    budget="ai2/oe-base",
    tasks=[TaskSpec(**task_spec_args)],
)

# Create the first experiment
experiment = b.experiment.create(spec=experiment_spec, workspace="ai2/olmocr")
print(f"Created benchmark experiment: {experiment.id}")
print(f"View at: https://beaker.org/ex/{experiment.id}")
print("-------")
print("")

# Second experiment: Performance test
perf_commands = []
if has_aws_creds:
    perf_commands.extend([
        "mkdir -p ~/.aws",
        'echo "$AWS_CREDENTIALS_FILE" > ~/.aws/credentials'
    ])

# Shell script for performance test
perf_shell = dedent("""\
set -euo pipefail

# Start vllm server in background
echo "Starting vllm server for Chandra..."
vllm serve datalab-to/chandra --served-model-name chandra > /tmp/vllm_server.log 2>&1 &
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
time chandra /root/olmOCR-mix-0225_benchmark_set/ /root/olmOCR-mix-0225_benchmark_set_chandra --method vllm

# Kill vllm server
echo "Stopping vllm server..."
kill $VLLM_PID || true
wait $VLLM_PID 2>/dev/null || true
""")

perf_commands.extend([
    chandra_install_cmd,
    "pip install --upgrade vllm",  # Ensure vllm is installed
    "pip install awscli",
    "aws s3 cp --recursive s3://ai2-oe-data/jakep/olmocr/olmOCR-mix-0225/benchmark_set/ /root/olmOCR-mix-0225_benchmark_set/",
    f"bash -c '{perf_shell}'"
])

# Build performance task spec
perf_task_spec_args = {
    "name": "chandra-performance",
    "image": ImageSource(beaker=f"{beaker_user}/{image_tag}"),
    "command": [
        "bash", "-c",
        " && ".join(perf_commands)
    ],
    "context": TaskContext(
        priority=Priority.normal,
        preemptible=True,
    ),
    "resources": TaskResources(gpu_count=1),
    "constraints": Constraints(cluster=["ai2/ceres-cirrascale", "ai2/jupiter-cirrascale-2"]),
    "result": ResultSpec(path="/noop-results"),
}

# Add env vars if AWS credentials exist
if has_aws_creds:
    perf_task_spec_args["env_vars"] = [
        EnvVar(name="AWS_CREDENTIALS_FILE", secret=aws_creds_secret)
    ]

# Create performance experiment spec
perf_experiment_spec = ExperimentSpec(
    description=f"Chandra {chandra_version_label} Performance Test - Branch: {git_branch}, Commit: {git_hash}",
    budget="ai2/oe-base",
    tasks=[TaskSpec(**perf_task_spec_args)],
)

# Create the performance experiment
perf_experiment = b.experiment.create(spec=perf_experiment_spec, workspace="ai2/olmocr")
print(f"Created performance experiment: {perf_experiment.id}")
print(f"View at: https://beaker.org/ex/{perf_experiment.id}")
EOF

# Run the Python script to create the experiments
echo "Creating Beaker experiments..."
$PYTHON /tmp/run_benchmark_experiment.py $IMAGE_TAG $BEAKER_USER $GIT_BRANCH $GIT_HASH "$CHANDRA_VERSION" "$CHANDRA_INSTALL_CMD"

# Clean up temporary file
rm /tmp/run_benchmark_experiment.py

echo "Benchmark experiments submitted successfully!"