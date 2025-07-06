import argparse
import os
import sys
import numpy as np
import tensorflow as tf
import wave
import glob

# 禁用 TensorFlow 2.x 的 Eager Execution，以兼容您提供的脚本
tf.compat.v1.disable_eager_execution()

# 假设 input_data.py 和 models.py 在同一路径下
# 这些模块来自您提供的 TensorFlow Speech Commands 示例
try:
    import input_data
    import models
    # 'frontend_op' 是 'micro' 预处理所必需的
    from tensorflow.lite.experimental.microfrontend.python.ops import audio_microfrontend_op as frontend_op
except ImportError as e:
    print("="*50)
    print(f"错误：导入模块失败 - {e}")
    print("请确保此脚本与 TensorFlow Speech Commands 示例中的")
    print("input_data.py 和 models.py 文件位于同一目录下，")
    print("并且 TensorFlow 环境完整。")
    print("="*50)
    sys.exit(1)


# --- 从你的训练配置中获取的常量 ---
# 模型和标签设置
WANTED_WORDS = "hey_vibe"
PREPROCESS = 'micro'
# 音频参数
SAMPLE_RATE = 16000
CLIP_DURATION_MS = 1000
WINDOW_SIZE_MS = 30.0
WINDOW_STRIDE_MS = 20.0
FEATURE_BIN_COUNT = 40
# 量化参数
QUANT_INPUT_MIN = 0.0
QUANT_INPUT_MAX = 26.0
# 文件路径
MODELS_DIR = 'models'
MODEL_TFLITE = os.path.join(MODELS_DIR, 'model.tflite')
# 定义静音标签以供过滤
SILENCE_LABEL = '_silence_'


def load_wav_file(path):
    """加载 WAV 文件并返回 int16 格式的 numpy 数组"""
    try:
        with wave.open(path, 'rb') as wf:
            sample_rate = wf.getframerate()
            if sample_rate != SAMPLE_RATE:
                print(f"跳过文件（采样率错误）: {os.path.basename(path)} ({sample_rate}Hz), 需要 {SAMPLE_RATE}Hz")
                return None
            
            n_channels = wf.getnchannels()
            if n_channels != 1:
                print(f"跳过文件（非单声道）: {os.path.basename(path)} ({n_channels} channels), 需要 1 channel")
                return None
                
            pcm_data = wf.readframes(wf.getnframes())
            audio_array = np.frombuffer(pcm_data, dtype=np.int16)
            return audio_array
    except FileNotFoundError:
        print(f"错误：文件未找到 - {path}")
        return None
    except Exception as e:
        print(f"读取 '{os.path.basename(path)}' 时出错: {e}")
        return None

def get_features_for_raw_data(audio_data_float, model_settings):
    """为原始音频数据（float32 numpy 数组）生成特征。"""
    graph = tf.Graph()
    with graph.as_default():
        audio_placeholder = tf.compat.v1.placeholder(tf.float32, [None, 1], name='audio_data')
        sample_rate = model_settings['sample_rate']
        window_size_ms = (model_settings['window_size_samples'] * 1000) / sample_rate
        window_step_ms = (model_settings['window_stride_samples'] * 1000) / sample_rate
        int16_input = tf.cast(tf.multiply(audio_placeholder, 32767), tf.int16)
        micro_frontend = frontend_op.audio_microfrontend(
            int16_input, sample_rate=sample_rate, window_size=window_size_ms,
            window_step=window_step_ms, num_channels=model_settings['fingerprint_width'],
            out_scale=1, out_type=tf.float32
        )
        output_features = tf.multiply(micro_frontend, (10.0 / 256.0))

    with tf.compat.v1.Session(graph=graph) as sess:
        reshaped_audio = np.reshape(audio_data_float, (-1, 1))
        features = sess.run(output_features, feed_dict={audio_placeholder: reshaped_audio})
        
    return features

def process_wav_file(wav_path, interpreter, model_settings, words_list, args):
    """对单个WAV文件进行加载、预处理和推理。"""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 加载并处理音频
    audio_data_int16 = load_wav_file(wav_path)
    if audio_data_int16 is None:
        return # 如果加载失败，则跳过此文件

    silence_padding = np.zeros(SAMPLE_RATE, dtype=np.int16)
    padded_audio = np.concatenate([silence_padding, audio_data_int16, silence_padding])
    
    # 滑动窗口推理
    samples_per_window = int((CLIP_DURATION_MS / 1000) * SAMPLE_RATE)
    samples_per_step = int(samples_per_window * 0.2)
    total_steps = (len(padded_audio) - samples_per_window) // samples_per_step + 1
    
    print(f"--> 分析 '{os.path.basename(wav_path)}' (时长: {len(padded_audio)/SAMPLE_RATE:.2f}s, 总步数: {total_steps})")
    print("-" * 40)
    
    found_in_file = False
    for i in range(total_steps):
        start_index = i * samples_per_step
        end_index = start_index + samples_per_window
        audio_window_int16 = padded_audio[start_index:end_index]
        audio_window_float32 = audio_window_int16.astype(np.float32) / 32768.0

        fingerprint_features = get_features_for_raw_data(audio_window_float32, model_settings)
        fingerprint_flat = fingerprint_features.flatten()
        
        quantized_fingerprint = np.clip(
            (fingerprint_flat - QUANT_INPUT_MIN) * (255 / (QUANT_INPUT_MAX - QUANT_INPUT_MIN)) - 128,
            -128, 127
        ).astype(np.int8)
        
        input_tensor = np.expand_dims(quantized_fingerprint, axis=0)
        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        interpreter.invoke()
        
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]
        top_prediction_index = np.argmax(output_data)
        top_prediction_label = words_list[top_prediction_index]
        
        scale, zero_point = output_details[0]['quantization']
        top_prediction_score = (int(output_data[top_prediction_index]) - zero_point) * scale
        # 将 output_data 中的原始量化值转换为浮点分数
        output_data = [(int(val) - zero_point) * scale for val in output_data]
        timestamp_ms = int(start_index / SAMPLE_RATE * 1000)
        
        if top_prediction_label == WANTED_WORDS:
            print(f"✅ [ 时间: {timestamp_ms/1000 - len(silence_padding)/SAMPLE_RATE:.2f}s ] --- {words_list[0]} ({output_data[0]:.2f}) {words_list[1]} ({output_data[1]:.2f}) '{words_list[2]}' ({output_data[2]:.2f})")
            found_in_file = True
        else:
            print(f"   [ 时间: {timestamp_ms/1000 - len(silence_padding)/SAMPLE_RATE:.2f}s ] --- {words_list[0]} ({output_data[0]:.2f}) {words_list[1]} ({output_data[1]:.2f}) '{words_list[2]}' ({output_data[2]:.2f})")
    
    if not found_in_file:
        print(f"❌ 在此文件中未检测到 '{WANTED_WORDS}'。")


def main(args):
    # 1. 准备模型设置和标签 (只需一次)
    model_settings = models.prepare_model_settings(
        len(WANTED_WORDS.split(',')), SAMPLE_RATE, CLIP_DURATION_MS, WINDOW_SIZE_MS,
        WINDOW_STRIDE_MS, FEATURE_BIN_COUNT, PREPROCESS
    )
    words_list = input_data.prepare_words_list(WANTED_WORDS.split(','))

    # 2. 加载 TFLite 模型 (只需一次)
    if not os.path.exists(MODEL_TFLITE):
        print(f"错误：找不到量化模型文件 '{MODEL_TFLITE}'")
        sys.exit(1)
    interpreter = tf.lite.Interpreter(model_path=MODEL_TFLITE)
    interpreter.allocate_tensors()
    
    # 3. 查找所有 .wav 文件
    if not os.path.isdir(args.input_dir):
        print(f"错误：提供的路径不是一个有效的文件夹: {args.input_dir}")
        sys.exit(1)
        
    # 使用 glob 递归搜索所有 .wav 文件
    wav_files = glob.glob(os.path.join(args.input_dir, '**', '*.wav'), recursive=True)
    if not wav_files:
        print(f"在文件夹 '{args.input_dir}' 中没有找到 .wav 文件。")
        return
        
    print(f"\n在文件夹 '{args.input_dir}' 中找到了 {len(wav_files)} 个 .wav 文件，准备开始处理...")

    # 4. 依次处理每个文件
    for wav_path in wav_files:
        print("\n" + "="*50)
        process_wav_file(wav_path, interpreter, model_settings, words_list, args)
    
    print("\n" + "="*50)
    print("所有文件处理完毕。")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="对一个文件夹内所有 .wav 文件进行唤醒词检测。")
    parser.add_argument(
        '--input_dir', 
        type=str, 
        required=True, 
        help='包含 .wav 文件的文件夹路径。'
    )
    parser.add_argument(
        '--verbose', 
        action='store_true', 
        help='是否打印所有非静音的识别结果。'
    )
    
    args = parser.parse_args()
    main(args)