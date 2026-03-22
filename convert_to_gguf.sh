#!/bin/bash
# Convert DGPO-trained model to GGUF format
# Usage: bash convert_to_gguf.sh [quantization_type]

set -e

MODEL_DIR="./dgpo-qwen2.5-1.5b/dgpo_student"
OUTPUT_DIR="./gguf-output"
QUANT_TYPE="${1:-Q4_K_M}"  # Default: Q4_K_M

echo "=========================================="
echo "Converting DGPO model to GGUF"
echo "=========================================="

# Clone llama.cpp if not exists
if [ ! -d "llama.cpp" ]; then
    echo "Cloning llama.cpp..."
    git clone https://github.com/ggerganov/llama.cpp.git
fi

# Build llama.cpp
echo "Building llama.cpp..."
cd llama.cpp
mkdir -p build && cd build
cmake .. -DLLAMA_BUILD_SERVER=ON
cmake --build . --config Release -j$(nproc)
cd ../..

# Install conversion dependencies
echo "Installing conversion dependencies..."
pip install -r llama.cpp/requirements.txt

# Convert to GGUF (FP16 first)
echo "Converting to GGUF (FP16)..."
python llama.cpp/convert_hf_to_gguf.py \
    "${MODEL_DIR}" \
    --outtype f16 \
    --outfile "${OUTPUT_DIR}/dgpo-qwen2.5-1.5b-f16.gguf"

# Quantize
echo "Quantizing to ${QUANT_TYPE}..."
./llama.cpp/build/bin/llama-quantize \
    "${OUTPUT_DIR}/dgpo-qwen2.5-1.5b-f16.gguf" \
    "${OUTPUT_DIR}/dgpo-qwen2.5-1.5b-${QUANT_TYPE}.gguf" \
    "${QUANT_TYPE}"

echo "=========================================="
echo "Conversion complete!"
echo "Output: ${OUTPUT_DIR}/dgpo-qwen2.5-1.5b-${QUANT_TYPE}.gguf"
echo "=========================================="

# Print file size
ls -lh "${OUTPUT_DIR}/dgpo-qwen2.5-1.5b-${QUANT_TYPE}.gguf"
