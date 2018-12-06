import os
import argparse

import numpy as np
from scipy.io import wavfile
from hmmlearn import hmm
from python_speech_features import mfcc

# 解析命令行的输入参数
def build_arg_parser():
    parser = argparse.ArgumentParser(description='Trains the HMM classifier')
    parser.add_argument("--input-folder", dest="input_folder", required=True,
            help="Input folder containing the audio files in subfolders")
    return parser

# 创建类，处理HMM相关过程
class HMMTrainer(object):
    '''用到高斯隐马尔科夫模型
    n_components：定义了隐藏状态的个数
    cov_type：定义了转移矩阵的协方差类型
    n_iter:定义了训练的迭代次数
    '''
    def __init__(self, model_name='GaussianHMM', n_components=4, cov_type='diag', n_iter=1000):
        self.model_name = model_name
        self.n_components = n_components
        self.cov_type = cov_type
        self.n_iter = n_iter
        self.models = []

        if self.model_name == 'GaussianHMM':
            self.model = hmm.GaussianHMM(n_components=self.n_components,
                    covariance_type=self.cov_type, n_iter=self.n_iter)
        else:
            raise TypeError('Invalid model type')

    # X是二维数组，其中每一行有13个数
    def train(self, X):
        np.seterr(all='ignore')
        self.models.append(self.model.fit(X))

    # 对输入数据运行模型
    def get_score(self, input_data):
        return self.model.score(input_data)

if __name__=='__main__':
    # 解析输入参数
    args = build_arg_parser().parse_args()
    input_folder = args.input_folder

    hmm_models = [] # 初始化隐马尔科夫模型的变量

    # 解析输入路径
    for dirname in os.listdir(input_folder):
        # 获取子文件夹名称
        subfolder = os.path.join(input_folder, dirname)
        if not os.path.isdir(subfolder):
            continue
        # 子文件夹名称即为该类的标记
        # 提取特征
        label = subfolder[subfolder.rfind('/') + 1:]

        # 初始化变量
        X = np.array([])
        y_words = []

        # 迭代所有音频文件(分别保留一个进行测试)
        for filename in [x for x in os.listdir(subfolder) if x.endswith('.wav')][:-1]:
            # 读取每个音频文件
            filepath = os.path.join(subfolder, filename)
            sampling_freq, audio = wavfile.read(filepath)

            # 提取MFCC特征
            mfcc_features = mfcc(audio, sampling_freq)

            # 将MFCC特征添加到X变量
            if len(X) == 0:
                X = mfcc_features
            else:
                X = np.append(X, mfcc_features, axis=0)

            # 添加标记
            y_words.append(label)

        print('X.shape =', X.shape)
        # 训练并且保存HMM模型
        hmm_trainer = HMMTrainer()
        hmm_trainer.train(X)
        hmm_models.append((hmm_trainer, label))
        hmm_trainer = None

    # 测试文件
    input_files = [
            'data/pineapple/pineapple15.wav',
            'data/orange/orange15.wav',
            'data/apple/apple15.wav',
            'data/kiwi/kiwi15.wav'
            ]

    # 为输入数据分类
    for input_file in input_files:
        # 读取每个音频文件
        sampling_freq, audio = wavfile.read(input_file)

        # 提取MFCC特征
        mfcc_features = mfcc(audio, sampling_freq)

        # 定义变量
        max_score = None
        output_label = None

        # 迭代HMM模型并选取得分最高的模型
        for item in hmm_models:
            hmm_model, label = item
            score = hmm_model.get_score(mfcc_features)
            if score > max_score:
                max_score = score
                output_label = label

        # 打印结果
        print("\nTrue:", input_file[input_file.find('/')+1:input_file.rfind('/')])
        print("Predicted:", output_label)

