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

COPY pyproject.toml README.md ./

# ① pip 系をアップデート
# ② 先に mamba-ssm を単独で入れる（失敗箇所を切り分けやすくする）
# ③ 最後に自分のパッケージをインストール
RUN pip install --upgrade pip setuptools wheel \
 && pip install "mamba-ssm[causal-conv1d]" \
 && pip install ".[dev]"

CMD ["/bin/bash"]
