FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libglib2.0-0 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install yq CLI
RUN wget -q https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64 -O /usr/bin/yq && \
    chmod +x /usr/bin/yq

# Install Orion and HeartWise_StatPlots
RUN git clone https://github.com/HeartWise-AI/Orion.git /workspace/Orion && \
    git clone https://github.com/HeartWise-AI/HeartWise_StatPlots.git /workspace/HeartWise_StatPlots

# Install Python dependencies
RUN pip install --no-cache-dir \
    numpy \
    scipy \
    pandas \
    pillow \
    matplotlib \
    seaborn \
    scikit-learn \
    scikit-image \
    pydicom \
    pylibjpeg \
    pylibjpeg-libjpeg \
    opencv-python-headless \
    imageio \
    imageio-ffmpeg \
    moviepy==1.0.2 \
    av \
    ffmpeg-python \
    timm \
    transformers>=4.30.0 \
    wandb \
    pyyaml \
    tqdm \
    tabulate \
    fvcore \
    torchmetrics \
    timesformer_pytorch \
    stam_pytorch

RUN pip install --no-cache-dir -e /workspace/Orion && \
    pip install --no-cache-dir -e /workspace/HeartWise_StatPlots

# Pre-download and verify official DeepCORO-CLIP Stenosis weights (455 MB)
RUN mkdir -p /workspace/weights/deepcoro_clip_generic/zcb8cu0l_20260213-033432 && \
    curl -f -L -sS --retry 5 --retry-delay 3 --retry-all-errors \
      -o /workspace/weights/deepcoro_clip_generic/zcb8cu0l_20260213-033432/best_model_epoch_11.pt \
      https://huggingface.co/heartwise/deepcoro_clip_generic/resolve/main/zcb8cu0l_20260213-033432/best_model_epoch_11.pt && \
    echo "4eac7676bc24930466d84f04f813fcbc1b9e70b056232cad603f18c84d008267  /workspace/weights/deepcoro_clip_generic/zcb8cu0l_20260213-033432/best_model_epoch_11.pt" | sha256sum -c -

# Pre-download official VasoVision preprocessing weights (44 MB)
RUN mkdir -p /workspace/weights/VasoVision && \
    curl -f -L -sS --retry 5 --retry-delay 3 --retry-all-errors \
      -o /workspace/weights/VasoVision/vaso_vision.pt \
      https://huggingface.co/heartwise/VasoVision/resolve/main/vaso_vision.pt

# Copy source repository and master runner
COPY config/ /workspace/config/
COPY dataloaders/ /workspace/dataloaders/
COPY models/ /workspace/models/
COPY projects/ /workspace/projects/
COPY runners/ /workspace/runners/
COPY scripts/ /workspace/scripts/
COPY utils/ /workspace/utils/
COPY docs/ /workspace/docs/
COPY run_deepcoro.py /workspace/run_deepcoro.py

ENV PYTHONPATH="/workspace:/workspace/Orion:$PYTHONPATH"

# Default help command
CMD ["python", "/workspace/run_deepcoro.py", "--help"]
