#!/bin/bash

# Create benchmark runner script
cat > run_benchmark.sh << 'EOL'
#!/bin/bash

# Set base directory (default to current directory if not specified)
BASE_DIR="${1:-.}"

echo "1. Creating directory structure under: $BASE_DIR"
mkdir -p "${BASE_DIR}/benchmark/configs" || exit 1
mkdir -p "${BASE_DIR}/benchmark/scripts" || exit 1
mkdir -p "${BASE_DIR}/benchmark/results" || exit 1

echo "2. Directory structure created:"
tree "${BASE_DIR}/benchmark"

echo "3. Running MoE benchmark script..."
python examples/moe/benchmark_moe.py \
    --configs-dir "${BASE_DIR}/benchmark/configs" \
    --scripts-dir "${BASE_DIR}/benchmark/scripts" \
    --pending-csv "${BASE_DIR}/benchmark/results/pending_experiments2.csv" \
    --benchmark-csv "${BASE_DIR}/benchmark/results/bench_final2.csv" \
    --base-config examples/config_tiny_llama_bench.yaml \
    --partition hopper-prod \
    --time 01:00:00 \
    --run

echo "4. Benchmark jobs submitted! Results will be saved to:"
echo "   - Configs: ${BASE_DIR}/benchmark/configs"
echo "   - Results: ${BASE_DIR}/benchmark/results"
EOL

# Make the script executable
chmod +x run_benchmark.sh

echo "Setup complete! Run the benchmark with:"
echo "./run_benchmark.sh [optional-base-directory]"
