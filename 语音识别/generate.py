import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

# 定义存储音频的输出文件
output_file = 'output_generated.wav'

# 指定音频生成的参数
duration = 3            # 单位秒
sampling_freq = 44100   # 单位Hz
tone_freq = 587         # 音调的频率
min_val = -2 * np.pi
max_val = 2 * np.pi

# 生成音频信号
t = np.linspace(min_val, max_val, duration * sampling_freq)
audio = np.sin(2 * np.pi * tone_freq * t)

# 添加噪声(duration * sampling_freq个(0,1]之间的值)
noise = 0.4 * np.random.rand(duration * sampling_freq)
audio += noise

scaling_factor = pow(2,15) - 1  # 转换为16位整型数
audio_normalized = audio / np.max(np.abs(audio))    # 归一化
audio_scaled = np.int16(audio_normalized * scaling_factor)  # 这句话什么意思

write(output_file, sampling_freq, audio_scaled) # 写入输出文件

audio = audio[:100]

x_values = np.arange(0, len(audio), 1) / float(sampling_freq)
x_values *= 1000    # 将时间轴单位转换为秒

plt.plot(x_values, audio, color='black')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')
plt.title('Audio signal')
plt.show()
