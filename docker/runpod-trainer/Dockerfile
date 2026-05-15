# RunPod LoRA-trainer image for rosetta-bone storyteller.
#
# Base: RunPod's stock PyTorch image with CUDA 12.4 + Python 3.12.
# We pin dependencies for reproducibility — bumping any of these
# requires a new image tag (and a bump of train.remote.image in
# config/default.toml).
FROM runpod/pytorch:2.4.0-py3.12-cuda12.4.1-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1

# Pin to a known-good set. Update in lockstep with train.py and
# bump the image's CalVer tag.
RUN pip install --no-cache-dir \
        "transformers==4.46.3" \
        "peft==0.13.2" \
        "trl==0.12.1" \
        "accelerate==1.1.1" \
        "bitsandbytes==0.44.1" \
        "datasets==3.1.0" \
        "safetensors==0.4.5" \
        "boto3==1.35.71" \
        "hf-transfer==0.1.8"

WORKDIR /app
COPY train.py /app/train.py

# Bake the image tag in so the produced adapter's metadata.json
# can record exactly which trainer built it. Override at build time:
#   docker build --build-arg IMAGE_TAG=2026.05.15 ...
ARG IMAGE_TAG=dev
ENV IMAGE_TAG=${IMAGE_TAG}

ENTRYPOINT ["python", "/app/train.py"]
