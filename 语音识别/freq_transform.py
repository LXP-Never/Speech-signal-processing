import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

sampling_freq, audio = wavfile.read('input_freq.wav')   # 读取文件

audio = audio / np.max(audio)   # 归一化，标准化

len_audio = len(audio)  # 3251

# 应用傅里叶变换
transformed_signal = np.fft.fft(audio)
print(transformed_signal)
# [-0.04022912+0.j         -0.04068997-0.00052721j -0.03933007-0.00448355j
#  ... -0.03947908+0.00298096j -0.03933007+0.00448355j -0.04068997+0.00052721j]
half_length = int(np.ceil((len_audio + 1) / 2.0))   # np.ceil向上取整(向大的方向取整)
transformed_signal = abs(transformed_signal[0:half_length])
print(transformed_signal)
# [0.04022912 0.04069339 0.0395848  ... 0.08001755 0.09203427 0.12889393]
transformed_signal /= float(len_audio)
transformed_signal **= 2

# 提取转换信号的长度
len_ts = len(transformed_signal)    # 1626

# 将部分信号乘以2
if len_audio % 2:   # 奇数
    transformed_signal[1:len_ts] *= 2
else:               # 偶数
    transformed_signal[1:len_ts-1] *= 2

# 获取功率信号
power = 10 * np.log10(transformed_signal)

# 建立时间轴
x_values = np.arange(0, half_length, 1) * (sampling_freq / len_audio) / 1000.0

# 绘制语音信号的
plt.figure()
plt.plot(x_values, power, color='blue')
plt.xlabel('Freq (in kHz)')
plt.ylabel('Power (in dB)')
plt.show()

