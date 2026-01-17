FROM tensorflow/tensorflow:2.13.0-gpu

WORKDIR /workspace

# System dependencies (for OpenCV, etc.)
RUN apt-get update && apt-get install -y \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copys only requirements (caching optimization)
COPY requirements.txt .

# Installs Python deps (no TensorFlow here)
RUN pip install --no-cache-dir -r requirements.txt

# Copys the entire project
COPY . .

# Shows default behavior: open shell (DVC controls the execution)
CMD ["bash"]
