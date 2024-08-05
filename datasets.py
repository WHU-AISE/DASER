import configs
import os

os.environ["CUDA_VISIBLE_DEVICES"] = getattr(configs, 'config_model')()['gpu_ids']

import pandas as pd
import torch
from torch.utils.data import Dataset


class Datasets(Dataset):
    def __init__(self, path):
        super(Datasets, self).__init__()

        # 加载数据集
        self.dataset = pd.read_feather(path)

        # 运行设备
        self.device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

        # 数据集长度
        self.len = self.dataset.shape[0]

    def __getitem__(self, i):
        # 取数据
        data = self.dataset.iloc[i]
        question = torch.as_tensor(data['question'], dtype=torch.int64, device=self.device)
        api_sequence = torch.as_tensor(data['api_sequence'], dtype=torch.int64, device=self.device)

        api_description_list_list = data['api_description'].tolist()
        api_description_list = []
        for api_description in api_description_list_list:
            api_description_list.append(torch.as_tensor(api_description, dtype=torch.int64, device=self.device))
        api_description_list.append(torch.as_tensor([2], dtype=torch.int64, device=self.device))

        # 返回数据
        output = {
            "question": question,
            "api_description": api_description_list,
            "api_sequence": api_sequence
        }
        return output

    def __len__(self):
        return self.len
