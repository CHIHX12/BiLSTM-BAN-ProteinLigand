# ============================================================================
# teiban (GCN-BiLSTM-BAN / DrugBAN model) GPU image.
#
# NOTE: On the current build host, "/" is ~98% full, so prefer building the
# .sif directly with `bash build_sif.sh` (uses /home for scratch). This
# Dockerfile is for machines that DO have disk room for Docker, or when you
# want a Docker image to push to a registry.
#
# Build:   docker build -t teiban:gpu .
# Run:     docker run --rm -it --gpus all teiban:gpu                # GPU self-check
#          docker run --rm -it --gpus all teiban:gpu python3 /opt/teiban/predict_simple.py
# To .sif: singularity build teiban.sif docker-daemon://teiban:gpu
# ============================================================================
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    TEIBAN_HOME=/opt/teiban

# OS deps (Python 3.10 ships with ubuntu 22.04)
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev \
        ca-certificates wget \
        libxrender1 libxext6 libsm6 libgl1 \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --no-cache-dir --upgrade pip

# PyTorch 2.2.1 + CUDA 12.1
RUN python3 -m pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu121 \
        torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1

# DGL 2.1.0 + CUDA 12.1  (cu121 wheel lives under wheels/cu121/, not wheels/repo.html)
# torchdata pinned to 0.7.1: dgl 2.1.0 imports torchdata.datapipes (removed in >=0.8).
# pydantic is an undeclared dgl 2.1.0 graphbolt dependency (import fails without it).
RUN python3 -m pip install --no-cache-dir \
        --find-links https://data.dgl.ai/wheels/cu121/repo.html \
        dgl==2.1.0+cu121 torchdata==0.7.1 "pydantic<3"

# Rest of the scientific / cheminformatics stack (numpy pinned last -> 1.26.4)
RUN python3 -m pip install --no-cache-dir \
        dgllife==0.3.2 \
        numpy==1.26.4 pandas==2.3.3 scikit-learn==1.7.0 scipy==1.15.2 matplotlib==3.10.1 \
        rdkit==2024.3.5 \
        yacs==0.1.8 prettytable==3.17.0 tqdm==4.67.1

WORKDIR /opt/teiban
# .dockerignore controls what is excluded (datasets, FDA_drugs, screening/output, .git ...)
COPY . /opt/teiban/
RUN rm -rf /opt/teiban/screening/output \
    && find /opt/teiban -type d -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true \
    && chmod -R a+rX /opt/teiban

CMD ["python3", "-c", "import torch; print('torch', torch.__version__, '| CUDA build:', torch.version.cuda, '| GPU available:', torch.cuda.is_available()); print('Run: python3 /opt/teiban/predict_simple.py')"]
