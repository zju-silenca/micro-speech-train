# test_directory.py
import tensorflow as tf
import numpy as np
from scipy.io import wavfile
import model_settings as ms
import argparse  # 用于接收命令行参数
import glob      # 用于查找文件
import os        # 用于处理文件路径

# --- 1. 加载模型和标签 ---
# MODEL_PATH = 'models/model.tflite'
MODEL_PATH = 'models/model.tflite'
LABELS_PATH = 'labels.txt'

# 加载标签
try:
    with open(LABELS_PATH, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    print(f"Error: {LABELS_PATH} not found. Please make sure the labels file is in the same directory.")
    exit()

# 加载TFLite模型并分配张量
try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
except Exception as e:
    print(f"Error loading model {MODEL_PATH}: {e}")
    exit()


# 获取输入和输出张量的详细信息
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

# 打印模型期望的输入形状，用于调试
print("Model Expected Input Shape:", input_details['shape'])

# --- 2. 音频预处理函数 (与之前相同) ---
def preprocess_audio(wav_path):
    sample_rate, audio_data = wavfile.read(wav_path)

    if sample_rate != ms.SAMPLE_RATE:
        # 简单重采样，对于高质量测试可能需要更复杂的库如librosa
        # 但对于基础测试，这通常足够了
        resampling_ratio = ms.SAMPLE_RATE / sample_rate
        new_length = int(len(audio_data) * resampling_ratio)
        audio_data = np.interp(
            np.linspace(0.0, len(audio_data) - 1, new_length),
            np.arange(len(audio_data)),
            audio_data
        ).astype(np.int16)
        # print(f"Warning: Resampled {wav_path} from {sample_rate}Hz to {ms.SAMPLE_RATE}Hz")


    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1).astype(np.int16)

    audio_data_float = audio_data.astype(np.float32) / np.iinfo(np.int16).max

    if len(audio_data_float) < ms.EXPECTED_SAMPLES:
        audio_data_float = np.pad(audio_data_float, (0, ms.EXPECTED_SAMPLES - len(audio_data_float)), 'constant')
    else:
        audio_data_float = audio_data_float[:ms.EXPECTED_SAMPLES]

    stfts = tf.signal.stft(audio_data_float,
                           frame_length=ms.WINDOW_SIZE_SAMPLES,
                           frame_step=ms.WINDOW_STRIDE_SAMPLES,
                           fft_length=ms.WINDOW_SIZE_SAMPLES)
    spectrograms = tf.abs(stfts)

    num_spectrogram_bins = stfts.shape[-1]
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        ms.FEATURE_BIN_COUNT, num_spectrogram_bins, ms.SAMPLE_RATE, 20, 4000)
    mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)
    log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
    features = log_mel_spectrograms

    # ##########################
    # ### 修改的关键部分在这里 ###
    # ##########################
    #
    # 移除之前的 tf.expand_dims 调用。
    # 根据报错，模型期望一个2D输入，最常见的形式是 (1, num_features)。
    # 我们将 (49, 40) 的频谱图压平成一维，然后增加一个批次维度。
    #
    # 获取模型期望的输入形状
    expected_shape = input_details['shape']
    
    # 将2D特征 (49, 40) 压平
    flattened_features = tf.reshape(features, [-1])
    
    # 调整为模型期望的形状，通常是 (1, 1960)
    # expected_shape[0] 通常是1 (批次大小), expected_shape[1] 是特征总数
    final_features = tf.reshape(flattened_features, expected_shape)

    input_scale, input_zero_point = input_details["quantization"]
    quantized_features = final_features / input_scale + input_zero_point
    quantized_features = quantized_features.numpy().astype(input_details["dtype"])

    return quantized_features

# --- 3. 运行推理的函数 (已修改返回) ---
def predict(wav_path):
    input_data = preprocess_audio(wav_path)
    if input_data is None:
        return None, None, None # 文件读取失败

    input_data = preprocess_audio(wav_path)
    interpreter.set_tensor(input_details['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details['index'])[0]
    output_scale, output_zero_point = output_details["quantization"]
    dequantized_output = (output_data.astype(np.float32) - output_zero_point) * output_scale
    predicted_index = np.argmax(dequantized_output)
    predicted_label = labels[predicted_index]
    confidence = dequantized_output[predicted_index]

    # **修改点：返回包含所有分数的数组**
    return predicted_label, confidence, dequantized_output

# --- 主程序 (已修改为处理目录) ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test a wake word model on all WAV files in a directory.")
    parser.add_argument("directory_path", type=str, help="Path to the directory containing .wav files to test.")
    args = parser.parse_args()

    # 查找所有.wav文件
    wav_files_path = os.path.join(args.directory_path, '*.wav')
    wav_files = glob.glob(wav_files_path)

    if not wav_files:
        print(f"No .wav files found in '{args.directory_path}'.")
        exit()

    print(f"Found {len(wav_files)} WAV files to test.")
    print("=" * 40)

    # 循环测试每个文件
    for wav_file in wav_files:
        try:
            # **修改点：接收函数返回的第三个值**
            label, conf, all_scores = predict(wav_file)

            # 打印结果
            print(f"File: {os.path.basename(wav_file)}")
            print(f"--> Predicted label: '{label}' with confidence {conf:.3f}")

            # **修改点：使用接收到的 all_scores 变量**
            print("    Detailed scores:")
            for i, probability in enumerate(all_scores):
                 print(f"      - {labels[i]:<15}: {probability:.3f}")
            print("-" * 20)

        except Exception as e:
            print(f"Could not process file {os.path.basename(wav_file)}: {e}")
            print("-" * 20)