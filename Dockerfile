FROM python:3.11-slim

ARG TORCH_VERSION=2.10.0+cpu
ARG TORCHVISION_VERSION=0.25.0+cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    AIRFLOW_HOME=/workspace/airflow_home \
    PYTHONPATH=/workspace/src

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md TRAINING.md requirements-train.txt ./
COPY src ./src
COPY scripts ./scripts
COPY configs ./configs
COPY main.py ./main.py

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --index-url https://download.pytorch.org/whl/cpu \
        "torch==${TORCH_VERSION}" \
        "torchvision==${TORCHVISION_VERSION}" \
    && python -m pip install -e ".[model]" \
    && python -m pip install -r requirements-train.txt \
    && python -m pip uninstall -y opencv-python \
    && python -m pip install --force-reinstall --no-deps "opencv-python-headless>=4.10,<5.1"

# The reproducible receipt fixture generator needs a Korean font in Linux jobs.
RUN apt-get update \
    && apt-get install -y --no-install-recommends fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

CMD ["python", "scripts/train_cashlog_category_from_uecfood.py"]
