FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install Python and basic libs
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3.11-venv python3-pip git \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install CUDA runtime libs needed for TF GPU
RUN apt-get update && apt-get install -y \
    libcudnn8 libcudnn8-dev \
    libcublas-12-0 \
    libcusparse-dev-12 \
    libcusolver-dev-12 \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

WORKDIR /workspace

COPY requirements.txt .

# Install Python dependencies + GPU-enabled TF 2.17.0 + DVC
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt && \
    python -m pip install --no-cache-dir --extra-index-url https://pypi.ngc.nvidia.com tensorflow==2.17.0 && \
    python -m pip install --no-cache-dir "dvc[s3]"

COPY . .

CMD ["/bin/bash"]
