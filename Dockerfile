# syntax=docker/dockerfile:1.7
#
# meiwakuhoryu/anima-comfyui:cuda13.0-wmd
#
# Target:
#   - RunPod
#   - NVIDIA RTX 3090 24GB (Ampere, sm_86)
#   - MiniMax H3 FL2VA INT8 Pruned ConvRot
#
# Philosophy:
#   - "everything included", but runtime features are switchable by ENV
#   - CUDA build toolchain is intentionally retained for custom-node/CUDA extension builds
#   - models/cache/input/output/user/custom_nodes live under /workspace
#   - Civitai FL2VA INT8 Pruned is the default diffusion model
#   - official Comfy-Org/Hugging Face is used for companion files and as fallback
#
# Build:
#   docker build -f Dockerfile.cuda13.0-wmd-minimax-h3 \
#     -t meiwakuhoryu/anima-comfyui:cuda13.0-wmd .
#
# For reproducibility, pin COMFYUI_REF to a tested commit SHA.
#

ARG BASE_IMAGE=nvidia/cuda:13.0.3-devel-ubuntu24.04
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG DEBIAN_FRONTEND=noninteractive

# PyTorch official cu130 pair.
ARG PYTORCH_VERSION=2.12.1
ARG TORCHVISION_VERSION=0.27.1
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu130

ARG COMFYUI_REPO=https://github.com/Comfy-Org/ComfyUI.git
ARG COMFYUI_REF=master

# RTX 3090 = Ampere sm_86.
# Override at build time for other cards:
#   RTX 4090 -> 8.9
#   3090+4090 universal build -> "8.6;8.9"
ARG TORCH_CUDA_ARCH_LIST=8.6

ARG INSTALL_SAGEATTENTION=1
ARG SAGEATTENTION_VERSION=2.2.0

ARG INSTALL_KJNODES=1
ARG KJNODES_REPO=https://github.com/kijai/ComfyUI-KJNodes.git
ARG KJNODES_REF=main

ARG INSTALL_WAVESPEED=1
ARG WAVESPEED_REPO=https://github.com/chengzeyi/Comfy-WaveSpeed.git
ARG WAVESPEED_REF=main

# Optional compatibility backends. Disabled because H3+3090 primarily uses SageAttention.
ARG INSTALL_XFORMERS=0
ARG INSTALL_FLASH_ATTN=0

ENV TZ=Asia/Tokyo \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_HOME=/usr/local/cuda \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    HF_HOME=/workspace/.cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/workspace/.cache/huggingface/hub \
    XDG_CACHE_HOME=/workspace/.cache \
    TORCH_HOME=/workspace/.cache/torch \
    TRITON_CACHE_DIR=/workspace/.cache/triton

# Full runtime + compiler/toolchain retained intentionally.
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-dev python3-venv python3-pip \
      build-essential gcc g++ make cmake ninja-build pkg-config \
      git git-lfs curl wget aria2 rsync jq ca-certificates \
      ffmpeg libsndfile1 libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
      libgomp1 libopenmpi-dev \
      unzip p7zip-full \
      procps pciutils lsof iproute2 net-tools \
      nano vim-tiny less \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install --system \
    && python3 -m venv /opt/venv \
    && /opt/venv/bin/python -m pip install --upgrade pip setuptools wheel

# PyTorch CUDA 13.0.
RUN python -m pip install \
      "torch==${PYTORCH_VERSION}" \
      "torchvision==${TORCHVISION_VERSION}" \
      --index-url "${PYTORCH_INDEX_URL}" \
    && python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
assert torch.version.cuda and torch.version.cuda.startswith("13."), torch.version.cuda
PY

# General model/video/audio/download/build helpers.
RUN python -m pip install \
      huggingface_hub \
      hf-xet \
      safetensors \
      accelerate \
      packaging \
      psutil \
      requests \
      aiohttp \
      numpy \
      pillow \
      scipy \
      soundfile \
      av \
      imageio \
      imageio-ffmpeg

# ComfyUI.
WORKDIR /opt
RUN git clone --filter=blob:none "${COMFYUI_REPO}" ComfyUI \
    && cd /opt/ComfyUI \
    && git fetch --tags --force \
    && git checkout "${COMFYUI_REF}" \
    && python -m pip install -r requirements.txt \
    && python - <<'PY'
import torch
print("After ComfyUI requirements:", torch.__version__, torch.version.cuda)
assert torch.version.cuda and torch.version.cuda.startswith("13."), \
    "ComfyUI requirements replaced the CUDA 13 PyTorch build"
PY

# SageAttention 2.2.
# Soft fallback is deliberate: an upstream SageAttention/PyTorch ABI change must not
# make the entire image impossible to build. Runtime automatically falls back to CK/SDPA.
RUN if [[ "${INSTALL_SAGEATTENTION}" == "1" ]]; then \
      set +e; \
      export CUDA_HOME=/usr/local/cuda \
             TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
             EXT_PARALLEL=4 \
             MAX_JOBS="$(nproc)" \
             NVCC_APPEND_FLAGS="--threads 8"; \
      python -m pip install "sageattention==${SAGEATTENTION_VERSION}" --no-build-isolation; \
      rc=$?; \
      if [[ $rc -ne 0 ]]; then \
        echo "WARNING: PyPI SageAttention build failed; trying current upstream source."; \
        rm -rf /tmp/SageAttention; \
        git clone --depth 1 https://github.com/thu-ml/SageAttention.git /tmp/SageAttention; \
        cd /tmp/SageAttention; \
        python -m pip install . --no-build-isolation; \
        rc=$?; \
      fi; \
      if [[ $rc -ne 0 ]]; then \
        echo "WARNING: SageAttention could not be built. Image will use Comfy Kitchen/PyTorch attention fallback."; \
      fi; \
      set -e; \
    fi

# Optional xFormers fallback for non-H3 workflows.
RUN if [[ "${INSTALL_XFORMERS}" == "1" ]]; then \
      python -m pip install xformers || true; \
    fi

# Optional FlashAttention. Not the preferred path on RTX 3090/H3.
RUN if [[ "${INSTALL_FLASH_ATTN}" == "1" ]]; then \
      python -m pip install flash-attn --no-build-isolation || true; \
    fi

# Preload optional/custom nodes outside ComfyUI.
# They are activated by symlink at runtime so they can be disabled without rebuilding.
RUN mkdir -p /opt/optional-nodes \
    && if [[ "${INSTALL_KJNODES}" == "1" ]]; then \
         git clone --filter=blob:none "${KJNODES_REPO}" /opt/optional-nodes/ComfyUI-KJNodes \
         && cd /opt/optional-nodes/ComfyUI-KJNodes \
         && git checkout "${KJNODES_REF}" \
         && if [[ -f requirements.txt ]]; then python -m pip install -r requirements.txt; fi; \
       fi \
    && if [[ "${INSTALL_WAVESPEED}" == "1" ]]; then \
         git clone --filter=blob:none "${WAVESPEED_REPO}" /opt/optional-nodes/Comfy-WaveSpeed \
         && cd /opt/optional-nodes/Comfy-WaveSpeed \
         && git checkout "${WAVESPEED_REF}" \
         && if [[ -f requirements.txt ]]; then python -m pip install -r requirements.txt || true; fi; \
       fi

# Ensure optional node requirements did not replace cu130 PyTorch.
RUN python - <<'PY'
import torch
print("Final torch:", torch.__version__)
print("Final torch CUDA:", torch.version.cuda)
assert torch.version.cuda and torch.version.cuda.startswith("13."), \
    "A custom-node dependency replaced the CUDA 13 PyTorch installation"
PY

# Persistent RunPod paths.
RUN mkdir -p \
      /workspace/input \
      /workspace/output \
      /workspace/temp \
      /workspace/user \
      /workspace/comfyui-models/diffusion_models \
      /workspace/comfyui-models/text_encoders \
      /workspace/comfyui-models/vae \
      /workspace/comfyui-models/loras \
      /workspace/custom_nodes \
      /workspace/.cache/huggingface \
      /workspace/.cache/torch \
      /workspace/.cache/triton

# ---------------------------------------------------------------------------
# Model downloader
#
# Default H3 diffusion:
#   Civitai modelVersionId=3193337, fileId=3074134
#   FL2VA INT8 Pruned ConvRot
#   SHA256 e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a
#
# Companion files:
#   Comfy-Org/MiniMax-H3 on Hugging Face
#
# Supported user ENV model syntax:
#   hf:OWNER/REPO@REVISION::path/to/file.safetensors
#   https://...
# separated with semicolons.
# ---------------------------------------------------------------------------
RUN cat >/usr/local/bin/model-sync.py <<'PY'
#!/usr/bin/env python3
from __future__ import annotations

import concurrent.futures
import hashlib
import os
import shutil
import sys
import urllib.parse
from pathlib import Path

import requests
from huggingface_hub import hf_hub_download

ROOT = Path(os.getenv("MODEL_ROOT", "/workspace/comfyui-models"))
TIMEOUT = int(os.getenv("MODEL_DOWNLOAD_TIMEOUT", "1800"))

DEFAULT_H3 = os.getenv("H3_AUTO_DOWNLOAD", "1") == "1"

CIVITAI_FL2VA_URL = os.getenv(
    "H3_FL2VA_URL",
    "https://civitai.red/api/download/models/3193337?fileId=3074134"
)
CIVITAI_FL2VA_NAME = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
CIVITAI_FL2VA_SHA256 = os.getenv(
    "H3_FL2VA_SHA256",
    "e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a"
)

HF_REPO = os.getenv("H3_HF_REPO", "Comfy-Org/MiniMax-H3")
HF_REV = os.getenv("H3_HF_REVISION", "main")

ENV_TO_DIR = {
    "CHECKPOINTS": "checkpoints",
    "DIFFUSION_MODELS": "diffusion_models",
    "UNETS": "unet",
    "TEXT_ENCODERS": "text_encoders",
    "CLIPS": "clip",
    "CLIP_VISIONS": "clip_vision",
    "VAES": "vae",
    "LORAS": "loras",
    "CONTROLNETS": "controlnet",
    "T2I_ADAPTERS": "controlnet",
    "UPSCALE_MODELS": "upscale_models",
    "EMBEDDINGS": "embeddings",
    "STYLE_MODELS": "style_models",
}

def split_specs(value: str):
    return [x.strip() for x in value.split(";") if x.strip()]

def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def verify(path: Path, expected: str | None):
    if not expected:
        return
    got = sha256(path)
    if got.lower() != expected.lower():
        raise RuntimeError(f"SHA256 mismatch for {path.name}: got {got}, expected {expected}")
    print(f"[models] SHA256 OK: {path.name}")

def http_download(url: str, dst: Path, expected_sha: str | None = None):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        if expected_sha and os.getenv("VERIFY_MODEL_HASHES", "1") == "1":
            verify(dst, expected_sha)
        else:
            print(f"[models] exists: {dst}")
        return

    part = dst.with_suffix(dst.suffix + ".part")
    start = part.stat().st_size if part.exists() else 0
    headers = {}
    if start:
        headers["Range"] = f"bytes={start}-"

    # Optional Civitai auth without baking secrets into the image.
    civitai_token = os.getenv("CIVITAI_TOKEN", "").strip()
    if civitai_token and "civitai." in url:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}token={urllib.parse.quote(civitai_token)}"

    with requests.get(
        url,
        headers=headers,
        stream=True,
        allow_redirects=True,
        timeout=(30, TIMEOUT),
    ) as r:
        if r.status_code == 416 and part.exists():
            part.replace(dst)
        else:
            r.raise_for_status()
            append = r.status_code == 206 and start > 0
            mode = "ab" if append else "wb"
            with part.open(mode) as f:
                for chunk in r.iter_content(chunk_size=16 * 1024 * 1024):
                    if chunk:
                        f.write(chunk)
            part.replace(dst)

    if expected_sha and os.getenv("VERIFY_MODEL_HASHES", "1") == "1":
        verify(dst, expected_sha)

def hf_download(repo: str, revision: str, repo_path: str, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        print(f"[models] exists: {dst}")
        return
    cached = hf_hub_download(
        repo_id=repo,
        filename=repo_path,
        revision=revision,
        token=os.getenv("HF_TOKEN") or None,
    )
    try:
        os.link(cached, dst)
    except OSError:
        shutil.copy2(cached, dst)

def download_spec(category: str, spec: str):
    target_dir = ROOT / category
    target_dir.mkdir(parents=True, exist_ok=True)

    if spec.startswith("hf:"):
        body = spec[3:]
        repo_rev, repo_path = body.split("::", 1)
        if "@" in repo_rev:
            repo, rev = repo_rev.rsplit("@", 1)
        else:
            repo, rev = repo_rev, "main"
        dst = target_dir / Path(repo_path).name
        print(f"[models] HF: {repo}@{rev}::{repo_path}")
        hf_download(repo, rev, repo_path, dst)
        return

    if spec.startswith("http://") or spec.startswith("https://"):
        name = Path(urllib.parse.urlparse(spec).path).name or "download.bin"
        dst = target_dir / name
        print(f"[models] HTTP: {spec}")
        http_download(spec, dst)
        return

    raise ValueError(f"Unsupported model spec: {spec}")

def download_default_h3():
    diffusion_dir = ROOT / "diffusion_models"
    text_dir = ROOT / "text_encoders"
    vae_dir = ROOT / "vae"

    diffusion = diffusion_dir / CIVITAI_FL2VA_NAME
    try:
        print("[models] MiniMax H3 FL2VA INT8 Pruned: Civitai")
        http_download(CIVITAI_FL2VA_URL, diffusion, CIVITAI_FL2VA_SHA256)
    except Exception as e:
        print(f"[models] Civitai failed: {e}", file=sys.stderr)
        print("[models] Falling back to official Comfy-Org/Hugging Face.")
        hf_download(
            HF_REPO, HF_REV,
            "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
            diffusion,
        )
        if os.getenv("VERIFY_MODEL_HASHES", "1") == "1":
            verify(diffusion, CIVITAI_FL2VA_SHA256)

    hf_download(
        HF_REPO, HF_REV,
        "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        text_dir / "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    )
    hf_download(
        HF_REPO, HF_REV,
        "vae/minimax_h3_video_vae_fp16.safetensors",
        vae_dir / "minimax_h3_video_vae_fp16.safetensors",
    )
    hf_download(
        HF_REPO, HF_REV,
        "vae/minimax_h3_audio_vae_fp32.safetensors",
        vae_dir / "minimax_h3_audio_vae_fp32.safetensors",
    )

def main():
    ROOT.mkdir(parents=True, exist_ok=True)

    if DEFAULT_H3:
        download_default_h3()

    jobs = []
    for env_name, category in ENV_TO_DIR.items():
        for spec in split_specs(os.getenv(env_name, "")):
            jobs.append((category, spec))

    if not jobs:
        return

    workers = max(1, int(os.getenv("MODEL_DOWNLOAD_CONCURRENCY", "3")))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(download_spec, category, spec) for category, spec in jobs]
        for f in concurrent.futures.as_completed(futures):
            f.result()

if __name__ == "__main__":
    main()
PY
RUN chmod +x /usr/local/bin/model-sync.py

# Runtime installer for additional custom nodes.
# Disabled by default; use explicit refs/SHAs where possible.
RUN cat >/usr/local/bin/custom-node-sync <<'BASH'
#!/usr/bin/env bash
set -Eeuo pipefail
[[ "${ALLOW_RUNTIME_CUSTOM_NODES:-0}" == "1" ]] || exit 0
[[ -n "${CUSTOM_NODE_REPOS:-}" ]] || exit 0

IFS=';' read -ra ITEMS <<< "${CUSTOM_NODE_REPOS}"
for item in "${ITEMS[@]}"; do
  [[ -n "${item}" ]] || continue
  repo="${item%%#*}"
  ref="main"
  [[ "${item}" == *"#"* ]] && ref="${item#*#}"
  name="$(basename "${repo%.git}")"
  dst="/workspace/custom_nodes/${name}"

  if [[ ! -d "${dst}/.git" ]]; then
    git clone --filter=blob:none "${repo}" "${dst}"
  fi

  (
    cd "${dst}"
    git fetch --all --tags --prune
    git checkout "${ref}"
    if [[ "${INSTALL_RUNTIME_CUSTOM_NODE_DEPS:-1}" == "1" && -f requirements.txt ]]; then
      python -m pip install -r requirements.txt
    fi
  )
done
BASH
RUN chmod +x /usr/local/bin/custom-node-sync

# ---------------------------------------------------------------------------
# Runtime launcher
# ---------------------------------------------------------------------------
RUN cat >/usr/local/bin/start-comfyui <<'BASH'
#!/usr/bin/env bash
set -Eeuo pipefail

COMFYUI_DIR="${COMFYUI_DIR:-/opt/ComfyUI}"
MODEL_ROOT="${MODEL_ROOT:-/workspace/comfyui-models}"
CUSTOM_NODES_DIR="${CUSTOM_NODES_DIR:-/workspace/custom_nodes}"

mkdir -p \
  "${MODEL_ROOT}" \
  "${CUSTOM_NODES_DIR}" \
  "${INPUT_DIR:-/workspace/input}" \
  "${OUTPUT_DIR:-/workspace/output}" \
  "${TEMP_DIR:-/workspace/temp}" \
  "${USER_DIR:-/workspace/user}" \
  "${HF_HOME:-/workspace/.cache/huggingface}" \
  "${TORCH_HOME:-/workspace/.cache/torch}" \
  "${TRITON_CACHE_DIR:-/workspace/.cache/triton}"

# Keep persistent user custom nodes without replacing the image's ComfyUI tree.
# Use symlinks for the preloaded optional nodes.
if [[ "${ENABLE_KJNODES:-1}" == "1" && -d /opt/optional-nodes/ComfyUI-KJNodes ]]; then
  ln -sfn /opt/optional-nodes/ComfyUI-KJNodes "${CUSTOM_NODES_DIR}/ComfyUI-KJNodes"
else
  rm -f "${CUSTOM_NODES_DIR}/ComfyUI-KJNodes" 2>/dev/null || true
fi

if [[ "${ENABLE_WAVESPEED:-0}" == "1" && -d /opt/optional-nodes/Comfy-WaveSpeed ]]; then
  ln -sfn /opt/optional-nodes/Comfy-WaveSpeed "${CUSTOM_NODES_DIR}/Comfy-WaveSpeed"
else
  rm -f "${CUSTOM_NODES_DIR}/Comfy-WaveSpeed" 2>/dev/null || true
fi

# ComfyUI normally loads <ComfyUI>/custom_nodes. Link persistent custom nodes.
if [[ -d "${COMFYUI_DIR}/custom_nodes" && ! -L "${COMFYUI_DIR}/custom_nodes" ]]; then
  mv "${COMFYUI_DIR}/custom_nodes" "${COMFYUI_DIR}/custom_nodes.image"
fi
ln -sfn "${CUSTOM_NODES_DIR}" "${COMFYUI_DIR}/custom_nodes"

/usr/local/bin/custom-node-sync

if [[ "${DOWNLOAD_MODELS:-1}" == "1" ]]; then
  /usr/local/bin/model-sync.py
fi

args=(
  --listen "${COMFYUI_LISTEN:-0.0.0.0}"
  --port "${COMFYUI_PORT:-8188}"
  --models-directory "${MODEL_ROOT}"
  --input-directory "${INPUT_DIR:-/workspace/input}"
  --output-directory "${OUTPUT_DIR:-/workspace/output}"
  --temp-directory "${TEMP_DIR:-/workspace/temp}"
  --user-directory "${USER_DIR:-/workspace/user}"
  --preview-method "${PREVIEW_METHOD:-none}"
  --max-upload-size "${MAX_UPLOAD_SIZE_MB:-2048}"
)

# Attention backend.
case "${ATTENTION_BACKEND:-sage}" in
  sage)
    if python -c 'import sageattention' >/dev/null 2>&1; then
      args+=(--use-sage-attention)
    else
      echo "[start] SageAttention unavailable; using Comfy Kitchen attention." >&2
      args+=(--use-ck-attention)
    fi
    ;;
  ck|comfy-kitchen) args+=(--use-ck-attention) ;;
  pytorch|sdpa) args+=(--use-pytorch-cross-attention) ;;
  flash) args+=(--use-flash-attention) ;;
  split) args+=(--use-split-cross-attention) ;;
  quad) args+=(--use-quad-cross-attention) ;;
  auto|"") ;;
  *) echo "Unknown ATTENTION_BACKEND=${ATTENTION_BACKEND}" >&2; exit 2 ;;
esac

case "${VRAM_MODE:-auto}" in
  auto|"") ;;
  high) args+=(--highvram) ;;
  low) args+=(--lowvram) ;;
  no|novram) args+=(--novram) ;;
  gpu|gpu-only) args+=(--gpu-only) ;;
  cpu) args+=(--cpu) ;;
  *) echo "Unknown VRAM_MODE=${VRAM_MODE}" >&2; exit 2 ;;
esac

case "${DYNAMIC_VRAM:-auto}" in
  1|on|true) args+=(--enable-dynamic-vram) ;;
  0|off|false) args+=(--disable-dynamic-vram) ;;
  auto|"") ;;
esac

case "${ASYNC_OFFLOAD:-auto}" in
  auto|"") ;;
  0|off|false) args+=(--disable-async-offload) ;;
  *) args+=(--async-offload "${ASYNC_OFFLOAD}") ;;
esac

[[ -n "${RESERVE_VRAM_GB:-}" ]] && args+=(--reserve-vram "${RESERVE_VRAM_GB}")
[[ "${FAST_DISK:-0}" == "1" ]] && args+=(--fast-disk)
[[ "${CUDA_GRAPHS:-1}" == "0" ]] && args+=(--disable-cuda-graphs)
[[ "${DISABLE_PINNED_MEMORY:-0}" == "1" ]] && args+=(--disable-pinned-memory)
[[ "${FORCE_NON_BLOCKING:-0}" == "1" ]] && args+=(--force-non-blocking)
[[ "${FP16_INTERMEDIATES:-0}" == "1" ]] && args+=(--fp16-intermediates)
[[ "${FORCE_CHANNELS_LAST:-0}" == "1" ]] && args+=(--force-channels-last)

case "${MMAP_MODE:-auto}" in
  on|1|true) args+=(--mmap-torch-files) ;;
  off|0|false) args+=(--disable-mmap) ;;
  auto|"") ;;
esac

# Cache. EasyCache is a workflow/node optimization and is not forced here.
case "${CACHE_MODE:-ram}" in
  ram)
    if [[ -n "${CACHE_RAM_GB:-}" ]]; then
      read -ra c <<< "${CACHE_RAM_GB}"
      args+=(--cache-ram "${c[@]}")
    else
      args+=(--cache-ram)
    fi
    ;;
  classic) args+=(--cache-classic) ;;
  lru) args+=(--cache-lru "${CACHE_LRU_SIZE:-10}") ;;
  none) args+=(--cache-none) ;;
  auto|"") ;;
esac

# Experimental ComfyUI fast features are opt-in.
# Example: FAST_FEATURES="fp16_accumulation cublas_ops autotune"
if [[ -n "${FAST_FEATURES:-}" ]]; then
  if [[ "${FAST_FEATURES}" == "all" ]]; then
    args+=(--fast)
  else
    read -ra f <<< "${FAST_FEATURES}"
    args+=(--fast "${f[@]}")
  fi
fi

# Future-proof escape hatch for new ComfyUI CLI switches.
if [[ -n "${COMFYUI_EXTRA_ARGS:-}" ]]; then
  read -ra extra <<< "${COMFYUI_EXTRA_ARGS}"
  args+=("${extra[@]}")
fi

echo "============================================================"
echo "MiniMax H3 / ComfyUI CUDA13 WMD"
echo "============================================================"
nvidia-smi || true
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("Compute capability:", torch.cuda.get_device_capability(0))
try:
    import sageattention
    print("SageAttention: available")
except Exception as e:
    print("SageAttention: unavailable:", e)
PY

printf '[start] python main.py'
printf ' %q' "${args[@]}"
printf '\n'

cd "${COMFYUI_DIR}"
exec python main.py "${args[@]}"
BASH
RUN chmod +x /usr/local/bin/start-comfyui

# Defaults are intentionally conservative; tune by ENV on RunPod.
ENV COMFYUI_DIR=/opt/ComfyUI \
    COMFYUI_LISTEN=0.0.0.0 \
    COMFYUI_PORT=8188 \
    MODEL_ROOT=/workspace/comfyui-models \
    INPUT_DIR=/workspace/input \
    OUTPUT_DIR=/workspace/output \
    TEMP_DIR=/workspace/temp \
    USER_DIR=/workspace/user \
    CUSTOM_NODES_DIR=/workspace/custom_nodes \
    DOWNLOAD_MODELS=1 \
    H3_AUTO_DOWNLOAD=1 \
    VERIFY_MODEL_HASHES=1 \
    MODEL_DOWNLOAD_CONCURRENCY=3 \
    MODEL_DOWNLOAD_TIMEOUT=1800 \
    H3_FL2VA_URL="https://civitai.red/api/download/models/3193337?fileId=3074134" \
    H3_FL2VA_SHA256=e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a \
    H3_HF_REPO=Comfy-Org/MiniMax-H3 \
    H3_HF_REVISION=main \
    ATTENTION_BACKEND=sage \
    VRAM_MODE=auto \
    DYNAMIC_VRAM=auto \
    ASYNC_OFFLOAD=auto \
    CACHE_MODE=ram \
    FAST_DISK=0 \
    CUDA_GRAPHS=1 \
    PREVIEW_METHOD=none \
    ENABLE_KJNODES=1 \
    ENABLE_WAVESPEED=0 \
    ALLOW_RUNTIME_CUSTOM_NODES=0 \
    INSTALL_RUNTIME_CUSTOM_NODE_DEPS=1 \
    MAX_UPLOAD_SIZE_MB=2048 \
    FAST_FEATURES= \
    COMFYUI_EXTRA_ARGS= \
    CHECKPOINTS= \
    DIFFUSION_MODELS= \
    UNETS= \
    TEXT_ENCODERS= \
    CLIPS= \
    CLIP_VISIONS= \
    VAES= \
    LORAS= \
    CONTROLNETS= \
    T2I_ADAPTERS= \
    UPSCALE_MODELS= \
    EMBEDDINGS= \
    STYLE_MODELS=

EXPOSE 8188

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=5 \
  CMD-SHELL curl -fsS "http://127.0.0.1:${COMFYUI_PORT:-8188}/system_stats" >/dev/null || exit 1

WORKDIR /opt/ComfyUI
ENTRYPOINT ["/usr/local/bin/start-comfyui"]
