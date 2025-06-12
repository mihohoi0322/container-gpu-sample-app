import tensorflow as tf
import numpy as np
import time
import os

def check_gpu():
    """GPU利用可能性をチェック"""
    print("=== GPU Check ===")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")
    
    if tf.config.list_physical_devices('GPU'):
        print("✅ GPU is available!")
        for gpu in tf.config.list_physical_devices('GPU'):
            print(f"GPU: {gpu}")
            # GPU詳細情報を表示
            gpu_details = tf.config.experimental.get_device_details(gpu)
            print(f"  Device name: {gpu_details.get('device_name', 'Unknown')}")
    else:
        print("❌ No GPU found, using CPU")
    
    # メモリ成長を有効化（GPUメモリエラー回避）
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✅ GPU memory growth enabled")
        except RuntimeError as e:
            print(f"❌ GPU configuration error: {e}")
    print()

def create_model():
    """簡単なニューラルネットワークモデルを作成"""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(512, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def generate_dummy_data(samples=10000):
    """ダミーデータを生成"""
    X = np.random.random((samples, 784)).astype(np.float32)
    y = np.random.randint(0, 10, samples)
    return X, y

def benchmark_training():
    """GPUとCPUのパフォーマンスを比較"""
    print("=== Training Benchmark ===")
    
    # GPUを明示的に指定
    device_name = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
    print(f"Using device: {device_name}")
    
    with tf.device(device_name):
        # データ準備
        X_train, y_train = generate_dummy_data(50000)
        X_test, y_test = generate_dummy_data(10000)
        
        # データをテンソルに変換
        X_train = tf.convert_to_tensor(X_train)
        y_train = tf.convert_to_tensor(y_train)
        X_test = tf.convert_to_tensor(X_test)
        y_test = tf.convert_to_tensor(y_test)
        
        # モデル作成
        model = create_model()
        
        # 訓練実行
        print("Starting training...")
        start_time = time.time()
        
        history = model.fit(
            X_train, y_train,
            batch_size=128,
            epochs=5,
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        end_time = time.time()
    
    training_time = end_time - start_time
    
    print(f"\n⏱️ Training completed in {training_time:.2f} seconds")
    
    # 評価
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"📊 Test accuracy: {test_accuracy:.4f}")
    
    return training_time, test_accuracy

def matrix_operations_benchmark():
    """行列演算のベンチマーク"""
    print("=== Matrix Operations Benchmark ===")
    
    # 大きな行列を作成
    size = 2000
    print(f"Creating {size}x{size} matrices...")
    
    with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
        # 行列作成
        matrix_a = tf.random.normal([size, size], dtype=tf.float32)
        matrix_b = tf.random.normal([size, size], dtype=tf.float32)
        
        # 行列乗算のベンチマーク
        print("Performing matrix multiplication...")
        start_time = time.time()
        
        result = tf.matmul(matrix_a, matrix_b)
        
        # 計算を実際に実行させる
        _ = result.numpy()
        
        end_time = time.time()
        
    operation_time = end_time - start_time
    print(f"⏱️ Matrix multiplication completed in {operation_time:.4f} seconds")
    
    return operation_time

def main():
    """メイン処理"""
    print("🚀 GPU Sample Application Starting...")
    print("=" * 50)
    
    # GPU確認とセットアップ
    check_gpu()
    
    # 実際に使用されるデバイスを確認
    print("=== Device Verification ===")
    with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
        test_tensor = tf.constant([1.0, 2.0, 3.0])
        print(f"Test tensor device: {test_tensor.device}")
    print()
    
    # 行列演算ベンチマーク
    matrix_time = matrix_operations_benchmark()
    print()
    
    # 機械学習ベンチマーク
    training_time, accuracy = benchmark_training()
    print()
    
    # 結果サマリー
    print("=" * 50)
    print("📈 RESULTS SUMMARY")
    print("=" * 50)
    print(f"Matrix Operations: {matrix_time:.4f} seconds")
    print(f"ML Training: {training_time:.2f} seconds")
    print(f"Final Accuracy: {accuracy:.4f}")
    
    device_type = "GPU" if tf.config.list_physical_devices('GPU') else "CPU"
    print(f"🔧 Executed on: {device_type}")
    print("✅ Application completed successfully!")

if __name__ == "__main__":
    main()