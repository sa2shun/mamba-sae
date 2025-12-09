# nvcc が入っている devel イメージに変更
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel

ENV LC_ALL=C.UTF-8 \
    LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential vim tmux htop \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# プロジェクト本体と mamba-ssm ソースをコピー
COPY pyproject.toml README.md ./
COPY mamba-src ./mamba-src

# ① pip 周りをアップデート
# ② jupyter / datasets をインストール
# ③ mamba-ssm を手元ソースからビルドし直す
# ④ 最後に自分のパッケージをインストール
RUN pip install --upgrade pip setuptools wheel \
 && pip install jupyterlab datasets \
 && cd /workspace/mamba-src \
 && rm -rf build mamba_ssm.egg-info selective_scan_cuda*.so \
 && pip install -e . --no-build-isolation \
 && cd /workspace \
 && pip install ".[dev]"

CMD ["/bin/bash"]
