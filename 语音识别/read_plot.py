import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# 读取音频文件
sampling_freq, audio = wavfile.read('input_read.wav')

# 打印参数
print('Shape:', audio.shape)        # Shape: (132300,) 说明有132300个值
print('数据类型:', audio.dtype)     # 数据类型: int16
# round(number, ndigits=0)对number进行四舍五入，ndigits如果是正数就是向右取整的位数
print('持续时间:', round(audio.shape[0] / float(sampling_freq), 3), 'seconds')
# 持续时间: 3.0 seconds
audio = audio / np.max(audio)    # 归一化

audio = audio[:30]  # 提取音频的前30个值

# 建立时间轴
x_values = np.arange(0, len(audio), 1) / float(sampling_freq)
x_values *= 1000    # 将单位转化为秒

# 画出声音信号图形
plt.plot(x_values, audio, color='blue')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')
plt.title('Audio signal')
plt.show()

