# test_padded_sliding_window.py
import tensorflow as tf
import numpy as np
from scipy.io import wavfile
import model_settings as ms
import argparse
import glob
import os
import warnings

# 忽略scipy读取wav文件时可能出现的元数据警告
warnings.filterwarnings("ignore", category=wavfile.WavFileWarning)

# --- 1. 全局设置 (模型、标签、解释器) ---
MODEL_PATH = 'models/model.tflite'
LABELS_PATH = 'labels.txt'

try:
    with open(LABELS_PATH, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    print(f"Error: {LABELS_PATH} not found.")
    exit()

try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
except Exception as e:
    print(f"Error loading model {MODEL_PATH}: {e}")
    exit()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

# --- 2. 核心处理函数 (处理一个1秒的音频块) ---
def process_and_predict(audio_chunk_float):
    """接收一个1秒的 float32 numpy 数组, 返回预测结果"""
    stfts = tf.signal.stft(audio_chunk_float,
                           frame_length=ms.WINDOW_SIZE_SAMPLES,
                           frame_step=ms.WINDOW_STRIDE_SAMPLES,
                           fft_length=ms.WINDOW_SIZE_SAMPLES)
    spectrograms = tf.abs(stfts)
    num_spectrogram_bins = stfts.shape[-1]
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        ms.FEATURE_BIN_COUNT, num_spectrogram_bins, ms.SAMPLE_RATE, 20, 4000)
    mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)
    features = tf.math.log(mel_spectrograms + 1e-6)

    expected_shape = input_details['shape']
    flattened_features = tf.reshape(features, [-1])
    final_features = tf.reshape(flattened_features, expected_shape)
    
    input_scale, input_zero_point = input_details["quantization"]
    quantized_features = final_features / input_scale + input_zero_point
    quantized_features = quantized_features.numpy().astype(input_details["dtype"])
    
    interpreter.set_tensor(input_details['index'], quantized_features)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details['index'])[0]
    
    output_scale, output_zero_point = output_details["quantization"]
    dequantized_output = (output_data.astype(np.float32) - output_zero_point) * output_scale
    
    return dequantized_output

# --- 主程序 (实现滑动窗口) ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test a wake word model on WAV files using a sliding window with silence padding.")
    parser.add_argument("directory_path", type=str, help="Path to the directory containing .wav files.")
    parser.add_argument("--stride_ms", type=int, default=250, help="Window stride in milliseconds.")
    parser.add_argument("--threshold", type=float, default=0.8, help="Confidence threshold for wake word detection.")
    args = parser.parse_args()

    wav_files_path = os.path.join(args.directory_path, '*.wav')
    wav_files = glob.glob(wav_files_path)
    if not wav_files:
        print(f"No .wav files found in '{args.directory_path}'.")
        exit()

    print(f"Found {len(wav_files)} WAV files. Stride: {args.stride_ms}ms, Threshold: {args.threshold}")
    print("=" * 40)

    for wav_file in wav_files:
        print(f"\nProcessing file: {os.path.basename(wav_file)}")
        try:
            sample_rate, audio_data = wavfile.read(wav_file)

            if sample_rate != ms.SAMPLE_RATE:
                print(f"Warning: Skipping file {os.path.basename(wav_file)} due to sample rate mismatch.")
                continue
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1).astype(np.int16)

            audio_float = audio_data.astype(np.float32) / np.iinfo(np.int16).max
            
            # ###############################################################
            # ### 修改点：在这里添加1秒的静音填充 ###
            # ###############################################################
            print("  Padding audio with 1s of silence at start and end...")
            padding_samples = ms.SAMPLE_RATE  # 1秒的采样点数
            silence_padding = np.zeros(padding_samples, dtype=np.float32)
            
            # 使用np.concatenate将 [静音, 原始音频, 静音] 拼接起来
            audio_float = np.concatenate([silence_padding, audio_float, silence_padding])
            # ###############################################################

            # 因为已经填充了静音，所以不再需要检查音频是否过短
            
            stride_samples = int(ms.SAMPLE_RATE * args.stride_ms / 1000)
            num_windows = (len(audio_float) - ms.EXPECTED_SAMPLES) // stride_samples + 1
            
            detected = False
            for i in range(num_windows):
                start_index = i * stride_samples
                end_index = start_index + ms.EXPECTED_SAMPLES
                audio_chunk = audio_float[start_index:end_index]
                
                all_scores = process_and_predict(audio_chunk)
                predicted_index = np.argmax(all_scores)
                predicted_label = labels[predicted_index]
                confidence = all_scores[predicted_index]
                
                # 计算时间戳时，要考虑前面的1秒填充
                # (可选) 如果你希望时间戳相对于原始音频，可以减去1000ms
                original_timestamp_ms = int(start_index * 1000 / ms.SAMPLE_RATE) - 1000
                print(f"  Window at {original_timestamp_ms}ms (relative to original): Label='{predicted_label}', Confidence={confidence:.3f}")
                
                if predicted_label not in ['_unknown_', '_silence_'] and confidence > args.threshold:
                    print(f"  ----> WAKE WORD DETECTED: '{predicted_label}' at ~{original_timestamp_ms}ms <----")
                    detected = True

            if not detected:
                print("  --> Wake word not detected in this file.")
                
        except Exception as e:
            print(f"  Error processing file {os.path.basename(wav_file)}: {e}")