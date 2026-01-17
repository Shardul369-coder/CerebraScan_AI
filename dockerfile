FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# System deps + Python
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3.11-venv python3-pip git \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Make python -> python3.11
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

WORKDIR /workspace

# Copy locked requirements first
COPY requirements.txt .

# Install deps + TensorFlow
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt && \
    python -m pip install --no-cache-dir tensorflow==2.17.0

# Optional: DVC
RUN python -m pip install --no-cache-dir "dvc[s3]"

# Copy all other project files
COPY . .

CMD ["/bin/bash"]
