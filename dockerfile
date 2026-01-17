FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# System deps
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3.11-venv python3-pip git \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy locked requirements first
COPY requirements.txt .

# Python deps
RUN python3.11 -m pip install --upgrade pip && \
    python3.11 -m pip install --no-cache-dir -r requirements.txt

# Optional: DVC if needed
RUN python3.11 -m pip install --no-cache-dir "dvc[s3]"

# Copy everything else
COPY . .

CMD ["/bin/bash"]
