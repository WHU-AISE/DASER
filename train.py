import os

import configs

os.environ["CUDA_VISIBLE_DEVICES"] = getattr(configs, 'config_model')()['gpu_ids']

import logging
import random
import time
from datetime import datetime
import coloredlogs
import numpy as np
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_scheduler
from module import Metric, DatasetTool, end_token
from datasets import Datasets
import torch
import argparse
import torch.distributed as dist
from model import Model
from apex import amp
from apex.parallel import DistributedDataParallel
from apex.parallel import convert_syncbn_model

class Trainer:
    def __init__(self):
        # 初始化config
        self.config_model = getattr(configs, 'config_model')()
        self.config_data = getattr(configs, 'config_data')()

        # 初始化logger
        self.logger = None
        self.tensorboard_writer = None
        self.timestamp = None
        self.init_logger()

        # 初始化随机数种子
        self.init_random_seed()

        # 加载预训练模型
        self.device = None
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.init_model()

        # 加载数据集
        self.train_loader = None
        self.test_loader = None
        self.evaluate_loader = None
        self.init_data()

        # 加载scheduler
        self.lr_scheduler = None
        self.init_scheduler()

    def init_logger(self):
        """
        初始化logger，记录训练信息
        :return:
        """
        self.logger = logging.getLogger(__name__)

        # 时间戳
        self.timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

        # 日志
        formatter = '%(asctime)s %(name)s[%(process)d] %(levelname)s %(message)s'
        coloredlogs.install(level='DEBUG', fmt=formatter)

        # 创建记录debug信息的file handler并添加到logger中
        os.makedirs(f"./output/{self.timestamp}/models", exist_ok=True)
        file_handler = logging.FileHandler(f"./output/{self.timestamp}/train_logs.txt")
        file_handler.setFormatter(logging.Formatter(formatter))
        self.logger.addHandler(file_handler)

        # tensorboard可视化显示 如果不存在目录则创建它
        os.makedirs(f"./output/{self.timestamp}/logs", exist_ok=True)
        self.tensorboard_writer = SummaryWriter(f"./output/{self.timestamp}/logs/")
        self.logger.info(f"Logger loading completed")

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
        self.logger.info(f"Random seed setting completed")

    def init_model(self):
        # 加载device
        self.device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
        self.model = Model.from_pretrained(self.config_data['model_path'])
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=3e-4)

        # 并行训练
        if torch.cuda.device_count() > 1 and self.config_model['parallel']:
            # 加载parser
            parser = argparse.ArgumentParser()
            parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
            args = parser.parse_args()

            dist.init_process_group(backend='nccl')
            torch.cuda.set_device(args.local_rank)

            # 同步BN
            self.model = convert_syncbn_model(self.model).to(self.device)
            # 混合精度
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level='O1')
            # 分布数据并行
            self.model = DistributedDataParallel(self.model, delay_allreduce=True)

            self.logger.info(f"Parallel training with multiple GPUs")
        else:
            self.model.to(self.device)
            self.logger.info(f"Using device: " + str(self.device))

        self.logger.info(f"Model loading completed")

    def init_data(self):
        """
        初始化data，加载数据集
        :return:
        """
        # 训练集
        self.train_loader = self.init_data_loader(self.config_data['train_path'], self.config_model['batch_size'])

        # 测试集
        self.test_loader = self.init_data_loader(self.config_data['test_path'], self.config_model['batch_size'])

        # 评估用
        self.evaluate_loader = self.init_data_loader(self.config_data['test_path'], 1)

        self.logger.info(f"Dataset loading completed")

    def init_data_loader(self, data_path, batch_size):
        """
        加载data_loader
        :param batch_size: batch大小
        :param data_path: 数据集path
        :return:
        """
        # 训练集
        data_set = Datasets(data_path)

        # 并行训练
        if torch.cuda.device_count() > 1 and self.config_model['parallel']:
            data_sampler = torch.utils.data.distributed.DistributedSampler(data_set)
            data_loader = DataLoader(dataset=data_set, batch_size=batch_size, collate_fn=DatasetTool.collate_fn,
                                     sampler=data_sampler)
        else:
            data_loader = DataLoader(dataset=data_set, batch_size=batch_size, shuffle=True,
                                     collate_fn=DatasetTool.collate_fn)

        return data_loader

    def init_scheduler(self):
        """
        加载scheduler
        :return:
        """
        # 加载lr_scheduler
        num_epochs = self.config_model['epoch']
        num_training_steps = num_epochs * len(self.train_loader)
        self.lr_scheduler = get_scheduler(name="linear", optimizer=self.optimizer, num_warmup_steps=0,
                                          num_training_steps=num_training_steps)

    def save_model(self, name, save_optimizer=False):
        """
        保存训练模型
        :return:
        """
        os.makedirs(f"./output/{self.timestamp}/models/{name}", exist_ok=True)
        if torch.cuda.device_count() > 1 and self.config_model['parallel']:
            self.model.module.save_pretrained(f"./output/{self.timestamp}/models/{name}")
        else:
            self.model.save_pretrained(f"./output/{self.timestamp}/models/{name}")

        if save_optimizer:
            torch.save(self.optimizer, f"./output/{self.timestamp}/models/{name}/optimizer.bin")

    def train(self, epoch):
        """
        模型训练
        :param epoch: epoch个数
        :return:
        """
        self.logger.info("Begin training")

        total_step = 0
        # 进行epoch个训练
        for i in range(epoch):
            self.model.train()
            step = 0
            step_num = self.train_loader.__len__()
            loss_list = []
            os.makedirs(f"./output/{self.timestamp}/models/{i}", exist_ok=True)

            # 进行每个step的训练
            for question_ids, question_mask, api_description_ids, api_description_mask, api_sequence_ids in tqdm(
                    self.train_loader, position=0, leave=True):
                start_time = time.time()
                step += 1
                total_step += 1

                # 开始训练
                self.model.train()

                # 清空之前的梯度
                self.optimizer.zero_grad()

                # 计算loss
                loss = self.model(input_ids=question_ids,
                                  attention_mask=question_mask,
                                  api_description_ids=api_description_ids,
                                  api_description_mask=api_description_mask,
                                  labels=api_sequence_ids,
                                  ).loss

                # 并行训练
                if torch.cuda.device_count() > 1 and self.config_model['parallel']:
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    # 反向传播, 获取新的梯度
                    loss.backward()

                # 用获取的梯度更新模型参数
                self.optimizer.step()

                # 调节学习率
                self.lr_scheduler.step()

                loss_list.append(loss.item())

                # 每隔100个step进行log输出
                if step % 100 == 0:
                    batch_time = time.time() - start_time
                    mean_loss = np.mean(loss_list)

                    # 绘制可视化
                    self.tensorboard_writer.add_scalar('loss', mean_loss, total_step)

                    log = 'epoch: [%d/%d], batch: [%d/%d], batch_time: %.2fs, loss: %.4f' % (
                        i + 1, epoch, step, step_num, batch_time, mean_loss)
                    loss_list = []
                    self.logger.info(log)

                # 每隔1000个step进行测试集验证 并绘制valid loss曲线
                if step % 1000 == 0:
                    self.logger.info("Begin validation")
                    self.model.eval()

                    # 记录测试集上的loss
                    valid_loss_list = []

                    # 在测试集上计算损失
                    for valid_question_ids, valid_question_mask, valid_api_description_ids, valid_api_description_mask, valid_api_sequence_ids in tqdm(
                            self.test_loader, position=0, leave=True):
                        # 不构建计算图
                        with torch.no_grad():
                            valid_loss = self.model(input_ids=valid_question_ids,
                                                    attention_mask=valid_question_mask,
                                                    api_description_ids=valid_api_description_ids,
                                                    api_description_mask=valid_api_description_mask,
                                                    labels=valid_api_sequence_ids,
                                                    ).loss

                        # 加入测试集
                        valid_loss_list.append(valid_loss.item())

                    # 平均验证loss
                    mean_valid_loss = np.mean(valid_loss_list)

                    # 绘制可视化
                    self.tensorboard_writer.add_scalar('valid_loss', mean_valid_loss, total_step)
                    self.logger.info(f"Validation loss: {mean_valid_loss}")

                    self.model.train()

            # epoch结束保存模型
            self.save_model(i)
            self.save_model("temp", True)

            # 进行评估 绘制bleu曲线
            self.logger.info("Begin evaluate")
            self.model.eval()
            bleu_list = []
            for question_ids, question_mask, api_description_ids, api_description_mask, api_sequence_ids in tqdm(
                    self.evaluate_loader, position=0, leave=True):
                # 不构建计算图
                with torch.no_grad():
                    if torch.cuda.device_count() > 1 and self.config_model['parallel']:
                        output_ids = self.model.module.greedy(input_ids=question_ids, attention_mask=question_mask)
                    else:
                        output_ids = self.model.greedy(input_ids=question_ids, attention_mask=question_mask)

                for output_index in range(len(output_ids)):
                    # 生成结果
                    output_id = output_ids[output_index]
                    n_output_id = []
                    for ids in output_id:
                        if end_token in ids:
                            n_output_id.append(ids[1:ids.index(end_token)])
                        else:
                            n_output_id.append(ids[1:])
                    output_id = n_output_id

                    # 标签
                    label_id = api_sequence_ids[output_index].tolist()
                    label_id = label_id[:label_id.index(end_token)]

                    # 计算bleu
                    bleu = Metric.calculate_bleu(output_id, label_id)
                    bleu_list.append(bleu)

            # 绘制可视化
            self.tensorboard_writer.add_scalar('bleu', np.mean(bleu_list) * 100, i + 1)
            self.model.train()

            # 打印训练信息
            self.logger.info("Epoch " + str(i + 1) + ", bleu is " + str(np.mean(bleu_list) * 100))

        # 最终结束保存模型
        self.save_model("final", True)
        self.logger.info("End training")


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train(getattr(configs, 'config_model')()['epoch'])
