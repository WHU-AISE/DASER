import configs
import os

os.environ["CUDA_VISIBLE_DEVICES"] = getattr(configs, 'config_model')()['gpu_ids']

import torch
from torch.utils.data import DataLoader
import random
import numpy as np
from datasets import Datasets
from evaluate import evaluate
from module import DatasetTool
from model import Model


class Tester:
    def __init__(self):
        # 初始化config
        self.config_data = getattr(configs, 'config_data')()
        self.config_model = getattr(configs, 'config_model')()

        # 初始化随机数种子
        self.init_random_seed()

        # 加载模型
        self.model = None
        self.init_model()

        # 加载测试集
        self.test_set = None
        self.test_loader = None
        self.init_data()

    def init_random_seed(self):
        """
        初始化随机数种子
        :return:
        """
        random.seed(self.config_model['random_seed'])
        os.environ['PYTHONHASHSEED'] = str(self.config_model['random_seed'])
        np.random.seed(self.config_model['random_seed'])
        torch.manual_seed(self.config_model['random_seed'])
        torch.cuda.manual_seed(self.config_model['random_seed'])
        torch.cuda.manual_seed_all(self.config_model['random_seed'])
        torch.backends.cudnn.deterministic = True

    def init_model(self):
        # 加载模型
        self.model = Model.from_pretrained(self.config_data['trained_model_path'])

        device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

    def init_data(self):
        # 测试集
        self.test_set = Datasets(self.config_data['test_path'])
        self.test_loader = DataLoader(dataset=self.test_set, batch_size=1, shuffle=False,
                                      collate_fn=DatasetTool.collate_fn)

    def test(self):
        """
        模型测试
        :return:
        """
        self.model.eval()

        evaluate(self.model, self.test_loader, output_path=self.config_data['trained_model_path'],
                 config_model=self.config_model, data_size=len(self.test_set))


if __name__ == '__main__':
    tester = Tester()
    tester.test()
