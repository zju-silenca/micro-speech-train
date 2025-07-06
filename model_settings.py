# model_settings.py
# 这些值基于 Colab Notebook 中 "Generate a TensorFlow Lite Model" 部分的权威定义

# --- 核心信号处理参数 ---
SAMPLE_RATE = 16000
CLIP_DURATION_MS = 1000
WINDOW_SIZE_MS = 30.0
WINDOW_STRIDE_MS = 20.0       # 这个值通常在 WINDOW_SIZE_MS 附近定义
FEATURE_BIN_COUNT = 40      # 频谱图的高度

# --- 模型架构 (确保与训练时选择的一致) ---
MODEL_ARCHITECTURE = 'tiny_conv'

# --- 根据核心参数计算得出的辅助常量 ---
# 以秒为单位的持续时间
DURATION_S = CLIP_DURATION_MS / 1000.0

# 期望的音频样本总数 (16000 * 1.0 = 16000)
EXPECTED_SAMPLES = int(SAMPLE_RATE * DURATION_S)

# 频谱图的宽度 (时间步长)
# 计算公式: (总时长 - 窗口大小) / 步长 + 1
# (1000 - 30) / 20 + 1 = 49.5 -> 49
FINGERPRINT_WIDTH = int((CLIP_DURATION_MS - WINDOW_SIZE_MS) / WINDOW_STRIDE_MS + 1)

# STFT窗口和步长的样本数
WINDOW_SIZE_SAMPLES = int(SAMPLE_RATE * WINDOW_SIZE_MS / 1000)
WINDOW_STRIDE_SAMPLES = int(SAMPLE_RATE * WINDOW_STRIDE_MS / 1000)