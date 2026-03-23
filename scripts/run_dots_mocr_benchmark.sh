#!/bin/bash

# Runs dots.mocr benchmark, measuring both olmOCR-bench performance and per document processing performance
# dots.mocr is served via vllm and called through the OpenAI-compatible API
# Usage:
#   ./scripts/run_dots_mocr_benchmark.sh                                      # Use default model and benchmark repo
#   ./scripts/run_dots_mocr_benchmark.sh --benchrepo allenai/olmOCR-bench-internal  # Use different benchmark repo
#   ./scripts/run_dots_mocr_benchmark.sh --benchbranch olmOCR-bench-1125      # Use specific branch/revision
#   ./scripts/run_dots_mocr_benchmark.sh --benchpath s3://ai2-oe-data/path/   # Use benchmark from S3 or local path
#   ./scripts/run_dots_mocr_benchmark.sh --cluster ai2/titan-cirrascale       # Specify a cluster
#   ./scripts/run_dots_mocr_benchmark.sh --beaker-image jakep/olmocr-benchmark-0.3.3-780bc7d934  # Skip Docker build
#   ./scripts/run_dots_mocr_benchmark.sh --noperf                             # Skip the performance test job
#   ./scripts/run_dots_mocr_benchmark.sh --model rednote-hilab/dots.mocr      # Use a specific model

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
cat << 'EOF' > /tmp/run_benchmark_experiment.py
import sys
import base64
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

# Default model for dots.mocr
dots_mocr_model = model if model else "rednote-hilab/dots.mocr"

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

# Python script that renders PDFs to images and calls dots.mocr via vllm OpenAI API
run_dots_mocr_python = dedent('''\
import os, sys, glob, json, base64, io
from openai import OpenAI
from pdf2image import convert_from_path

PDF_ROOT = "olmOCR-bench/bench_data/pdfs"
TARGET_ROOT = "olmOCR-bench/bench_data/dots_mocr"

# Clean and create target directory
if os.path.exists(TARGET_ROOT):
    import shutil
    shutil.rmtree(TARGET_ROOT)
os.makedirs(TARGET_ROOT, exist_ok=True)

client = OpenAI(base_url="http://localhost:8000/v1", api_key="empty")

PROMPT = """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: Text field should be omitted
    - Formula: Format as LaTeX
    - Table: Format as HTML
    - Others: Format as Markdown

4. Constraints:
    - Output text must be original from image, no translation
    - All layout elements sorted by reading order

5. Final Output: Single JSON object
"""

# Find all PDFs
pdf_files = sorted(glob.glob(os.path.join(PDF_ROOT, "**", "*.pdf"), recursive=True))
print(f"Found {len(pdf_files)} PDFs to process")

for pdf_path in pdf_files:
    rel_path = os.path.relpath(pdf_path, PDF_ROOT)
    parts = rel_path.split(os.sep)
    if len(parts) < 2:
        print(f"Warning: Unexpected PDF path layout for {pdf_path}, skipping")
        continue

    section = parts[0]
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]

    print(f"  Processing {pdf_path}...")
    try:
        # Render first page of PDF to image
        images = convert_from_path(pdf_path, first_page=1, last_page=1, dpi=144)
        if not images:
            print(f"  Warning: Could not render {pdf_path}")
            continue

        # Convert to base64
        buf = io.BytesIO()
        images[0].save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        # Call dots.mocr via OpenAI API
        response = client.chat.completions.create(
            model="model",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                        {"type": "text", "text": PROMPT},
                    ],
                }
            ],
            max_tokens=24000,
        )

        result_text = response.choices[0].message.content

        # Try to extract text content from JSON response and assemble markdown
        markdown_parts = []
        try:
            data = json.loads(result_text)
            elements = data if isinstance(data, list) else data.get("elements", data.get("layout", [data]))
            for elem in elements:
                text = elem.get("text", "")
                if text:
                    markdown_parts.append(text)
            if markdown_parts:
                result_text = "\n\n".join(markdown_parts)
        except (json.JSONDecodeError, TypeError, AttributeError):
            # If not valid JSON, use raw text as-is
            pass

        # Write output
        target_dir = os.path.join(TARGET_ROOT, section)
        os.makedirs(target_dir, exist_ok=True)
        target_path = os.path.join(target_dir, f"{pdf_name}_pg1_repeat1.md")

        with open(target_path, "w") as f:
            f.write(result_text)
        print(f"  Wrote {target_path}")
    except Exception as e:
        print(f"  Error processing {pdf_path}: {e}")
        continue

print("Done processing all PDFs")
''')

# Base64-encode the inference script so we can safely embed it in the command chain
import base64
run_dots_mocr_python_b64 = base64.b64encode(run_dots_mocr_python.encode()).decode()

# Shell script to start vllm, write inference script, run it, and clean up
run_dots_mocr_shell = dedent("""\
bash -lc 'set -euo pipefail

# Decode the inference script
echo "__INFERENCE_SCRIPT_B64__" | base64 -d > /tmp/run_dots_mocr_inference.py

# Start vllm server in background
echo "Starting vllm server for dots.mocr..."
vllm serve __DOTS_MOCR_MODEL__ --served-model-name model --trust-remote-code --chat-template-content-format string --gpu-memory-utilization 0.9 > /tmp/vllm_server.log 2>&1 &
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

# Run the Python inference script
python /tmp/run_dots_mocr_inference.py

# Kill vllm server
echo "Stopping vllm server..."
kill $VLLM_PID || true
wait $VLLM_PID 2>/dev/null || true
'""").replace("__DOTS_MOCR_MODEL__", dots_mocr_model).replace("__INFERENCE_SCRIPT_B64__", run_dots_mocr_python_b64)

# Check if HF_TOKEN secret exists
hf_token_secret = f"{beaker_user}-HF_TOKEN"
try:
    b.secret.get(hf_token_secret, workspace="ai2/olmocr")
    has_hf_token = True
    print(f"Found HuggingFace token secret: {hf_token_secret}")
except:
    has_hf_token = False
    print(f"HuggingFace token secret not found: {hf_token_secret}")

# First experiment: Benchmark job
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
commands.append("uv pip install --system s5cmd")

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

# Write the inference script to a temp file, install deps, run
commands.extend([
    "uv pip install --system --upgrade vllm",
    "uv pip install --system openai pdf2image",
    run_dots_mocr_shell,
    "python -m olmocr.bench.benchmark --dir ./olmOCR-bench/bench_data --candidate dots_mocr"
])

# Build task spec with optional env vars
if '/' in image_tag:
    image_ref = image_tag
else:
    image_ref = f"{beaker_user}/{image_tag}"

task_spec_args = {
    "name": "dots-mocr-benchmark",
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

# Create experiment spec
experiment_spec = BeakerExperimentSpec(
    description=f"dots.mocr Benchmark Run - Branch: {git_branch}, Commit: {git_hash}",
    budget="ai2/oe-base",
    tasks=[BeakerTaskSpec(**task_spec_args)],
)

# Create the experiment
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

    # Performance test Python script - processes PDFs from the benchmark set via vllm API
    perf_dots_mocr_python = dedent('''\
import os, sys, glob, json, base64, io, time
from openai import OpenAI
from pdf2image import convert_from_path

PDF_ROOT = "/root/olmOCR-mix-0225_benchmark_set"
TARGET_ROOT = "/root/olmOCR-mix-0225_benchmark_set_dots_mocr"

os.makedirs(TARGET_ROOT, exist_ok=True)

client = OpenAI(base_url="http://localhost:8000/v1", api_key="empty")

PROMPT = """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: Text field should be omitted
    - Formula: Format as LaTeX
    - Table: Format as HTML
    - Others: Format as Markdown

4. Constraints:
    - Output text must be original from image, no translation
    - All layout elements sorted by reading order

5. Final Output: Single JSON object
"""

pdf_files = sorted(glob.glob(os.path.join(PDF_ROOT, "**", "*.pdf"), recursive=True))
if not pdf_files:
    pdf_files = sorted(glob.glob(os.path.join(PDF_ROOT, "*.pdf")))
print(f"Found {len(pdf_files)} PDFs to process for performance test")

start_time = time.time()
for pdf_path in pdf_files:
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    print(f"  Processing {pdf_path}...")
    try:
        images = convert_from_path(pdf_path, first_page=1, last_page=1, dpi=144)
        if not images:
            continue
        buf = io.BytesIO()
        images[0].save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        response = client.chat.completions.create(
            model="model",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                        {"type": "text", "text": PROMPT},
                    ],
                }
            ],
            max_tokens=24000,
        )

        result_text = response.choices[0].message.content
        target_path = os.path.join(TARGET_ROOT, f"{pdf_name}.md")
        with open(target_path, "w") as f:
            f.write(result_text)
    except Exception as e:
        print(f"  Error processing {pdf_path}: {e}")
        continue

elapsed = time.time() - start_time
print(f"Performance test completed in {elapsed:.1f}s for {len(pdf_files)} PDFs")
''')

    # Base64-encode the perf script
    perf_dots_mocr_python_b64 = base64.b64encode(perf_dots_mocr_python.encode()).decode()

    # Shell script for performance test
    perf_shell = dedent("""\
set -euo pipefail

# Decode the perf script
echo "__PERF_SCRIPT_B64__" | base64 -d > /tmp/run_dots_mocr_perf.py

# Start vllm server in background
echo "Starting vllm server for dots.mocr..."
vllm serve __DOTS_MOCR_MODEL__ --served-model-name model --trust-remote-code --chat-template-content-format string --gpu-memory-utilization 0.9 > /tmp/vllm_server.log 2>&1 &
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
time python /tmp/run_dots_mocr_perf.py

# Kill vllm server
echo "Stopping vllm server..."
kill $VLLM_PID || true
wait $VLLM_PID 2>/dev/null || true
""").replace("__DOTS_MOCR_MODEL__", dots_mocr_model).replace("__PERF_SCRIPT_B64__", perf_dots_mocr_python_b64)

    perf_commands.extend([
        "pip install uv",
        "uv pip install --system --upgrade vllm",
        "uv pip install --system openai pdf2image awscli",
        "aws s3 cp --recursive s3://ai2-oe-data/jakep/olmocr/olmOCR-mix-0225/benchmark_set/ /root/olmOCR-mix-0225_benchmark_set/",
        f"bash -c '{perf_shell}'"
    ])

    # Build performance task spec
    perf_task_spec_args = {
        "name": "dots-mocr-performance",
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
        description=f"dots.mocr Performance Test - Branch: {git_branch}, Commit: {git_hash}",
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
CMD="$PYTHON /tmp/run_benchmark_experiment.py $IMAGE_TAG $BEAKER_USER $GIT_BRANCH $GIT_HASH"

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
rm /tmp/run_benchmark_experiment.py

echo "Benchmark experiments submitted successfully!"
