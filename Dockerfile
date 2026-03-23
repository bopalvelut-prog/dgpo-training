FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y git cmake build-essential python3 python3-pip && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy training code
COPY . .

# Multi-GPU support
ENV NCCL_DEBUG=INFO
ENV CUDA_VISIBLE_DEVICES=all
ENV DEEPSPEED_ALLREDUCE_BUCKET_SIZE=5e8
ENV DEEPSPEED_ZERO3_BUCKET_SIZE=5e8

# Default command
CMD ["bash", "train_all_sizes.sh"]
