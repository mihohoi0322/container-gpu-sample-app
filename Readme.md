# GPU Sample Application

TensorFlowを使用してGPUとCPUのパフォーマンスを比較するベンチマークアプリケーションです。GPU環境が利用可能な場合はGPUで、利用できない場合はCPUで自動的に動作します。

## 📋 目次

- [概要](#概要)
- [機能](#機能)
- [動作環境](#動作環境)
- [セットアップ](#セットアップ)
- [実行方法](#実行方法)
- [出力例](#出力例)
- [トラブルシューティング](#トラブルシューティング)

## 🎯 概要

このアプリケーションは以下の処理を実行して、GPU/CPUのパフォーマンスを測定します：

1. **GPU環境の確認**: TensorFlowがGPUを認識しているかチェック
2. **行列演算ベンチマーク**: 2000×2000の行列乗算の実行時間を測定
3. **機械学習ベンチマーク**: ニューラルネットワークの訓練時間と精度を測定

## ⚡ 機能

### GPU環境チェック
- TensorFlowバージョンの表示
- 利用可能なGPUデバイスの検出・表示
- GPUメモリ設定の最適化

### ベンチマーク機能
- **行列演算**: 大規模行列乗算の性能測定
- **機械学習**: ニューラルネットワーク訓練の性能測定
- **自動デバイス選択**: GPU利用可能時はGPU、不可時はCPUで自動実行

### モデル仕様
- 入力層: 784次元（28×28画像想定）
- 隠れ層1: 512ニューロン + ReLU + Dropout(0.2)
- 隠れ層2: 256ニューロン + ReLU + Dropout(0.2) 
- 出力層: 10クラス分類（Softmax）

## 💻 動作環境

### GPU環境での実行（推奨）
- **ハードウェア**: NVIDIA GPU（CUDA対応）
- **ソフトウェア**: 
  - Docker
  - NVIDIA Docker Runtime
  - NVIDIA GPU Driver

### CPU環境での実行
- **ハードウェア**: 任意のCPU
- **ソフトウェア**: Docker

### Windows環境での実行
- **CPU実行**: Windows PowerShell/コマンドプロンプトから直接実行可能
- **GPU実行**: 
  - NVIDIA GPU: WSL2環境内での実行を推奨
  - ARM GPU (Qualcomm Adreno等): TensorFlow標準版では非対応
  - 代替案: TensorFlow Liteの使用を推奨

## 🚀 セットアップ

### 1. リポジトリのクローン
```bash
git clone <repository-url>
cd gpu-sample-app
```

### 2. 必要ファイルの確認
```
gpu-sample-app/
├── Dockerfile
├── main.py
├── requirements.txt
└── README.md
```

### 3. GPU環境の確認（GPU使用時のみ）

**NVIDIA GPU Driverの確認:**
```bash
nvidia-smi
```

**NVIDIA Docker Runtimeの確認:**
```bash
docker info | grep nvidia
```

## 📦 実行方法

### Dockerイメージのビルド
```bash
# イメージをビルド
docker build -t gpu-sample-app .

# キャッシュなしでビルド（問題発生時）
docker build --no-cache -t gpu-sample-app .
```

### アプリケーションの実行

#### GPU環境での実行
```bash
# GPUを使用して実行
docker run --gpus all gpu-sample-app

# インタラクティブモードで実行（デバッグ用）
docker run --gpus all -it gpu-sample-app

# バックグラウンドで実行
docker run --gpus all -d --name gpu-app gpu-sample-app
```

#### CPU環境での実行
```bash
# CPUで実行
docker run gpu-sample-app

# インタラクティブモードで実行
docker run -it gpu-sample-app
```

### デバッグ用コマンド
```bash
# コンテナ内でbashシェルを起動
docker run --gpus all -it gpu-sample-app bash

# 実行中のコンテナに接続
docker exec -it gpu-app bash
```

## 📊 出力例

### GPU環境での実行例
```
🚀 GPU Sample Application Starting...
==================================================
=== GPU Check ===
TensorFlow version: 2.13.0
GPU Available: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
✅ GPU is available!
GPU: /physical_device:GPU:0
  Device name: NVIDIA GeForce RTX 3080

=== Device Verification ===
Test tensor device: /job:localhost/replica:0/task:0/device:GPU:0

=== Matrix Operations Benchmark ===
Creating 2000x2000 matrices...
Performing matrix multiplication...
⏱️ Matrix multiplication completed in 0.0050 seconds

=== Training Benchmark ===
Using device: /GPU:0
Starting training...
Epoch 1/5
391/391 [==============================] - 2s 4ms/step - loss: 2.3127 - accuracy: 0.1003
...
⏱️ Training completed in 8.45 seconds
📊 Test accuracy: 0.1063

==================================================
📈 RESULTS SUMMARY
==================================================
Matrix Operations: 0.0050 seconds
ML Training: 8.45 seconds
Final Accuracy: 0.1063
🔧 Executed on: GPU
✅ Application completed successfully!
```

### CPU環境での実行例
```
🚀 GPU Sample Application Starting...
==================================================
=== GPU Check ===
TensorFlow version: 2.13.0
GPU Available: []
❌ No GPU found, using CPU

=== Device Verification ===
Test tensor device: /job:localhost/replica:0/task:0/device:CPU:0

=== Matrix Operations Benchmark ===
Creating 2000x2000 matrices...
Performing matrix multiplication...
⏱️ Matrix multiplication completed in 0.0271 seconds

=== Training Benchmark ===
Using device: /CPU:0
Starting training...
Epoch 1/5
391/391 [==============================] - 4s 9ms/step - loss: 2.3127 - accuracy: 0.1003
...
⏱️ Training completed in 17.16 seconds
📊 Test accuracy: 0.1063

==================================================
📈 RESULTS SUMMARY
==================================================
Matrix Operations: 0.0271 seconds
ML Training: 17.16 seconds
Final Accuracy: 0.1063
🔧 Executed on: CPU
✅ Application completed successfully!
```

## ⚖️ GPU vs CPU パフォーマンス比較

| 項目 | GPU (例) | CPU (例) | 性能比 |
|------|----------|----------|--------|
| 行列演算 | 0.005秒 | 0.027秒 | 約5.4倍高速 |
| ML訓練 | 8.45秒 | 17.16秒 | 約2.0倍高速 |

*実際の性能は使用するハードウェアにより大きく異なります*

## 🔧 トラブルシューティング

### Windows GPU関連の問題

#### 1. ARM系GPU（Qualcomm Adreno等）での実行
Windows Surface等のARM系GPUは、TensorFlow標準版では対応していません。

**現在の対応状況:**
- ❌ TensorFlow GPU版: NVIDIA CUDA専用
- ✅ TensorFlow CPU版: 正常動作
- ✅ TensorFlow Lite: ARM GPU対応（部分的）

**推奨解決方法:**
```powershell
# CPU環境での実行（現在の方法）
docker run gpu-sample-app

# 結果: CPU性能測定として有効
```

#### 2. NVIDIA GPU搭載Windowsマシンの場合
```powershell
# WSL2環境でのGPU実行
wsl
docker run --gpus all gpu-sample-app
```

#### 3. Windows環境での性能測定の価値
ARM系CPUでの測定結果も有効なベンチマークです：
- ARM系プロセッサーの性能測定
- モバイル・エッジデバイスでの推論性能の参考
- クラウド環境との性能比較

### GPU関連の問題

#### 1. GPUが認識されない
```bash
# NVIDIA GPU Driverの確認
nvidia-smi

# NVIDIA Docker Runtimeの確認
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu20.04 nvidia-smi
```

#### 2. "could not select device driver" エラー
- NVIDIA Docker Runtimeがインストールされていない
- GPUドライバーが古い、または破損している

**解決方法:**
```bash
# NVIDIA Docker Runtimeのインストール
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

#### 3. メモリ不足エラー
GPU環境では大量のメモリを使用する場合があります。

**解決方法:**
- より小さなバッチサイズを使用
- モデルのサイズを縮小
- 他のGPUアプリケーションを終了

### ビルド関連の問題

#### 1. パッケージインストールエラー
```bash
# キャッシュをクリアして再ビルド
docker build --no-cache -t gpu-sample-app .
```

#### 2. タイムゾーン設定で停止
Dockerfileの環境変数設定が正しく行われているか確認してください。

### CPU環境での動作

GPU環境がない場合でも、このアプリケーションは自動的にCPUモードで動作します：

- GPU検出処理は失敗しますが、エラーではありません
- 全ての計算がCPUで実行されます
- 実行時間は長くなりますが、正常に完了します
- 結果サマリーに「Executed on: CPU」と表示されます

## 📝 ファイル構成

### main.py
- `check_gpu()`: GPU環境の確認
- `create_model()`: ニューラルネットワークモデルの作成
- `generate_dummy_data()`: テストデータの生成
- `benchmark_training()`: 機械学習ベンチマーク
- `matrix_operations_benchmark()`: 行列演算ベンチマーク
- `main()`: メイン処理

### requirements.txt
```
tensorflow==2.13.0
numpy==1.24.3
```

### Dockerfile
- NVIDIA CUDA 11.8.0 base image使用
- 必要なシステムパッケージのインストール
- Python環境のセットアップ
- TensorFlowとその依存関係のインストール

## 🎓 学習・研究用途

このアプリケーションは以下の用途に活用できます：

- **GPU/CPU性能比較**: 実際の数値でパフォーマンス差を確認
- **TensorFlow環境テスト**: GPU環境のセットアップ確認
- **ベンチマーク**: 異なるハードウェア環境での性能測定
- **教育**: 機械学習のGPU活用について学習

---

## 📄 ライセンス

MIT License

## 🤝 コントリビューション

Issue や Pull Request をお待ちしています。