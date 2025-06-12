# NVIDIA CUDA base image with Python
FROM nvidia/cuda:11.8.0-base-ubuntu20.04

# 環境変数を設定してインタラクティブな入力を避ける
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo

# システムパッケージを更新し、必要な依存関係をインストール
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    pkg-config \
    libhdf5-dev \
    libhdf5-serial-dev \
    build-essential \
    cmake \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    libffi-dev \
    libssl-dev \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# Pythonのシンボリックリンクを作成
RUN ln -s /usr/bin/python3 /usr/bin/python

# pipをアップグレード
RUN pip3 install --upgrade pip setuptools wheel

# 作業ディレクトリを設定
WORKDIR /app

# requirements.txtをコピー
COPY requirements.txt .

# Pythonパッケージをインストール
RUN pip3 install --no-cache-dir -r requirements.txt

# アプリケーションファイルをコピー
COPY . .

# 環境変数をリセット
ENV DEBIAN_FRONTEND=

# デフォルトコマンド
CMD ["python", "main.py"]