import argparse
import os
import sys
import numpy as np
import tensorflow as tf
import wave
import glob

# 禁用 TensorFlow 2.x 的一些行为以匹配脚本的TF1.x兼容风格
# 注意：对于新项目，推荐使用纯TF2.x代码，但此处为了与您之前的脚本兼容
tf.compat.v1.disable_eager_execution()

# --- 从你的训练配置中获取的常量 ---
# 音频参数
SAMPLE_RATE = 16000
CLIP_DURATION_MS = 1000
WINDOW_SIZE_MS = 30.0
WINDOW_STRIDE_MS = 20.0
FEATURE_BIN_COUNT = 40
WANTED_WORDS = "hey_vibe" # 确保与您的模型一致

# 定义静音和未知标签以供过滤
SILENCE_LABEL = '_silence_'
UNKNOWN_WORD_LABEL = '_unknown_'

MODELS_DIR = 'models'
MODEL_TFLITE = os.path.join(MODELS_DIR, 'float_model.tflite')

def load_wav_file(path):
    """加载WAV文件并返回int16格式的numpy数组，会进行格式检查。"""
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
    except Exception as e:
        print(f"读取 '{os.path.basename(path)}' 时出错: {e}")
        return None

def get_features_for_raw_data(audio_data_float, model_settings):
    """为原始音频数据（float32 numpy数组）生成特征。"""
    try:
        from tensorflow.lite.experimental.microfrontend.python.ops import audio_microfrontend_op as frontend_op
    except ImportError as e:
        print("="*50)
        print("错误: 无法导入 'audio_microfrontend_op'。")
        print("请确保您的TensorFlow环境中包含了此模块。")
        print(f"原始错误: {e}")
        print("="*50)
        sys.exit(1)

    graph = tf.Graph()
    with graph.as_default():
        audio_placeholder = tf.compat.v1.placeholder(tf.float32, [None, 1], name='audio_data')
        sample_rate = model_settings['sample_rate']
        window_size_ms = (model_settings['window_size_samples'] * 1000) / sample_rate
        window_step_ms = (model_settings['window_stride_samples'] * 1000) / sample_rate
        
        int16_input = tf.cast(tf.multiply(audio_placeholder, 32767), tf.int16)
        micro_frontend = frontend_op.audio_microfrontend(
            int16_input,
            sample_rate=sample_rate, window_size=window_size_ms,
            window_step=window_step_ms, num_channels=model_settings['fingerprint_width'],
            out_scale=1, out_type=tf.float32
        )
        output_features = tf.multiply(micro_frontend, (10.0 / 256.0))

    with tf.compat.v1.Session(graph=graph) as sess:
        reshaped_audio = np.reshape(audio_data_float, (-1, 1))
        features = sess.run(output_features, feed_dict={audio_placeholder: reshaped_audio})
    return features

def process_wav_file(wav_path, interpreter, model_settings, labels, threshold):
    """对单个WAV文件进行加载、预处理和推理。"""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    audio_data_int16 = load_wav_file(wav_path)
    if audio_data_int16 is None:
        return

    # 滑动窗口参数
    samples_per_window = int((CLIP_DURATION_MS / 1000) * SAMPLE_RATE)
    samples_per_step = int(samples_per_window * 0.2)
    
    # 填充音频以确保边缘的词也能被捕获
    padding = np.zeros(samples_per_window // 2, dtype=np.int16)
    padded_audio = np.concatenate([padding, audio_data_int16, padding])
    
    total_steps = (len(padded_audio) - samples_per_window) // samples_per_step + 1

    print(f"--> 分析 '{os.path.basename(wav_path)}' (时长: {len(audio_data_int16)/SAMPLE_RATE:.2f}s, 总步数: {total_steps})")
    print("-" * 40)
    
    found_in_file = False
    for i in range(total_steps):
        start_index = i * samples_per_step
        audio_window_int16 = padded_audio[start_index : start_index + samples_per_window]
        
        # 将数据从 int16 转换为 float32 [-1.0, 1.0]
        audio_window_float32 = audio_window_int16.astype(np.float32) / 32768.0

        # 生成浮点特征
        features_2d = get_features_for_raw_data(audio_window_float32, model_settings)
        features_flat = features_2d.flatten()

        # 准备输入张量 (注意现在是 float32)
        interpreter.set_tensor(input_details[0]['index'], np.expand_dims(features_flat, axis=0))
        interpreter.invoke()
        
        # 获取输出 (也是 float32)
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]
        
        top_index = np.argmax(output_data)
        top_label = labels[top_index]
        top_score = output_data[top_index]

        # 应用置信度阈值来判断是否触发
        # if top_label == WANTED_WORDS and top_score > threshold:
        #     timestamp_s = (start_index - len(padding)) / SAMPLE_RATE
        #     print(f"✅ [时间: {max(0, timestamp_s - 1):.2f}s] --- 检测到唤醒词: '{top_label}' (置信度: {top_score:.2f})")
        #     found_in_file = True
        timestamp_ms = (start_index - len(padding)) / SAMPLE_RATE * 1000
        if top_label == WANTED_WORDS:
            print(f"✅ [ 时间: {timestamp_ms/1000 - len(padding)/SAMPLE_RATE:.2f}s ] --- {labels[0]} ({output_data[0]:.2f}) {labels[1]} ({output_data[1]:.2f}) '{labels[2]}' ({output_data[2]:.2f})")
            found_in_file = True
        else:
            print(f"   [ 时间: {timestamp_ms/1000 - len(padding)/SAMPLE_RATE:.2f}s ] --- {labels[0]} ({output_data[0]:.2f}) {labels[1]} ({output_data[1]:.2f}) '{labels[2]}' ({output_data[2]:.2f})")
    
    if not found_in_file:
        print(f"❌ 在此文件中未检测到高于阈值({threshold})的唤醒词。")

def main(args):
    # 准备模型设置和标签
    try:
        from models import prepare_model_settings
        from input_data import prepare_words_list
    except ImportError:
        print("错误: 无法从 'models.py' 或 'input_data.py' 导入函数。")
        print("请确保这些文件与您的推理脚本位于同一目录。")
        sys.exit(1)

    model_settings = prepare_model_settings(
        len(WANTED_WORDS.split(',')), SAMPLE_RATE, CLIP_DURATION_MS, WINDOW_SIZE_MS,
        WINDOW_STRIDE_MS, FEATURE_BIN_COUNT, 'micro'
    )
    labels = prepare_words_list(WANTED_WORDS.split(','))
    
    # 准备 TFLite Delegate (如果需要)
    delegates = []
    if args.use_fp16:
        try:
            # XNNPACK delegate 通常是 TensorFlow Lite 的一部分
            delegates.append(tf.lite.load_delegate('libtensorflowlite_xnnpack_delegate.so'))
            print("信息: 已启用 XNNPACK FP16 delegate。")
        except (ValueError, OSError) as e:
            print(f"警告: 无法加载 XNNPACK delegate: {e}")
            print("将继续使用标准 CPU kernel。")

    # 加载 TFLite 模型并创建 Interpreter
    try:
        interpreter = tf.lite.Interpreter(model_path=MODEL_TFLITE, experimental_delegates=delegates)
        interpreter.allocate_tensors()
    except Exception as e:
        print(f"错误: 加载模型或创建 Interpreter 失败: {e}")
        sys.exit(1)

    # 查找所有 .wav 文件
    if not os.path.isdir(args.input_dir):
        print(f"错误: 提供的路径不是一个有效的文件夹: {args.input_dir}")
        sys.exit(1)
        
    wav_files = glob.glob(os.path.join(args.input_dir, '**', '*.wav'), recursive=True)
    if not wav_files:
        print(f"在文件夹 '{args.input_dir}' 中没有找到 .wav 文件。")
        return
        
    print(f"\n在文件夹 '{args.input_dir}' 中找到了 {len(wav_files)} 个 .wav 文件，准备开始处理...")

    # 依次处理每个文件
    for wav_path in wav_files:
        print("\n" + "="*60)
        process_wav_file(wav_path, interpreter, model_settings, labels, args.threshold)
    
    print("\n" + "="*60)
    print("所有文件处理完毕。")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="对文件夹内的.wav文件进行float32/float16唤醒词检测。")
    parser.add_argument(
        '--input_dir', type=str, required=True, help='包含 .wav 文件的文件夹路径。'
    )
    parser.add_argument(
        '--threshold', type=float, default=0.85, help='识别唤醒词的置信度阈值 (0.0 到 1.0)。'
    )
    parser.add_argument(
        '--use_fp16', action='store_true', help='使用 XNNPACK delegate 进行 FP16 加速推理。'
    )
    
    args = parser.parse_args()
    main(args)