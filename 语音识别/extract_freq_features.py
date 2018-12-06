import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank

# 读取输入音频文件
sampling_freq, audio = wavfile.read("input_freq.wav")

# 提取MFCC和滤波器组特征
mfcc_features = mfcc(audio, sampling_freq)
filterbank_features = logfbank(audio, sampling_freq)

print('\nMFCC:\n窗口数 =', mfcc_features.shape[0])
print('每个特征的长度 =', mfcc_features.shape[1])
print('\nFilter bank:\n窗口数 =', filterbank_features.shape[0])
print('每个特征的长度 =', filterbank_features.shape[1])

# 画出特征图，将MFCC可视化。转置矩阵，使得时域是水平的
mfcc_features = mfcc_features.T
plt.matshow(mfcc_features)
plt.title('MFCC')
# 将滤波器组特征可视化。转置矩阵，使得时域是水平的
filterbank_features = filterbank_features.T
plt.matshow(filterbank_features)
plt.title('Filter bank')

plt.show()
