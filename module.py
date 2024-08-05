import configs
import os

os.environ["CUDA_VISIBLE_DEVICES"] = getattr(configs, 'config_model')()['gpu_ids']

import math
import numpy as np
import pandas as pd
import torch
import configs
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import single_meteor_score
from rouge import Rouge

# 运行设备
device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

begin_token = 0
pad_token = 1
end_token = 2


class Metric:
    config_data = getattr(configs, 'config_data')()
    api_dataset = pd.read_feather(config_data['api_path'])
    api_data_count = api_dataset.set_index('index')['total_count'].to_dict()
    vocab_size = api_dataset.iloc[0]['index']
    rouge = Rouge()

    @staticmethod
    def calculate_bleu(output_ids, label_ids):
        """
        计算BLEU
        :param output_ids: 模型输出值
        :param label_ids: 标签
        :return:
        """
        if len(output_ids) == 0:
            return 0

        bleu_list = []
        for i in range(len(output_ids)):
            # 未生成任何api
            if len(output_ids[i]) == 0:
                bleu_list.append(0)
                continue

            # 计算bleu
            bleu_gram = sentence_bleu([output_ids[i]], label_ids,
                                      weights=[(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)])

            # 如果标签的长度小于gram数 则仅统计最大值为标签长度的gram bleu
            bleu_len = min(len(bleu_gram), len(label_ids))
            bleu = 0
            for j in range(bleu_len):
                bleu += bleu_gram[j]
            bleu /= bleu_len

            # 计算bp
            bp = min(1, pow(math.e, (1 - len(label_ids) / len(output_ids[i]))))

            # 向bleu中加入bp惩罚
            bleu_list.append(bleu * bp)

        return np.max(bleu_list)

    @staticmethod
    def calculate_meteor(output_ids, label_ids):
        """
        计算METEORs
        :param output_ids: 模型输出值
        :param label_ids: 标签
        :return:
        """
        if len(output_ids) == 0:
            return 0

        meteor_list = []
        for i in range(len(output_ids)):
            # 未生成任何api
            if len(output_ids[i]) == 0:
                meteor_list.append(0)
                continue

            # 计算bleu
            score = single_meteor_score(list(str(id) for id in output_ids[i]), list(str(id) for id in label_ids))

            meteor_list.append(score)

        return np.max(meteor_list)

    @staticmethod
    def calculate_rouge(output_ids, label_ids):
        """
        计算ROUGE
        :param output_ids: 模型输出值
        :param label_ids: 标签
        :return:
        """
        if len(output_ids) == 0:
            return 0

        # 转化为由空格分隔的字符串
        output_ids = list(" ".join(str(id) for id in output) for output in output_ids)
        label_ids = " ".join(str(id) for id in label_ids)

        rouge_list = []
        for i in range(len(output_ids)):
            # 未生成任何api
            if len(output_ids[i]) == 0:
                rouge_list.append(0)
                continue

            # 计算rouge
            score = Metric.rouge.get_scores(output_ids[i], label_ids, avg=True)['rouge-l']['f']

            rouge_list.append(score)

        return np.max(rouge_list)

    @staticmethod
    def calculate_levenshtein_distance(output_ids):
        """
        计算levenshtein距离
        对于推荐结果多样性的度量，衡量Top-K结果中API序列两两之间的相似度，表示单个推荐结果中列表的多样性。ILS越大，代表推荐结果的多样性越好
        :param output_ids:
        :return:
        """
        # 输出序列个数
        seq_length = len(output_ids)
        if seq_length <= 1:
            return 0

        # 两两之间计算levenshtein距离
        levenshtein_list = []
        for i in range(seq_length):
            for j in range(i + 1, seq_length):
                levenshtein_list.append(Metric.levenshtein(output_ids[i], output_ids[j]))

        return (2 / (seq_length * (seq_length - 1))) * np.sum(levenshtein_list)

    @staticmethod
    def levenshtein(seq1, seq2):
        """
        levenshtein相似度度量
        :param seq1:
        :param seq2:
        :return:
        """
        len1 = len(seq1)
        len2 = len(seq2)

        dp = np.zeros((len1 + 1, len2 + 1), dtype=int)
        for i in range(len1 + 1):
            dp[i, 0] = i

        for i in range(len2 + 1):
            dp[0, i] = i

        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                t = 1
                if seq1[i - 1] == seq2[j - 1]:
                    t = 0

                dp[i][j] = min(dp[i - 1, j - 1] + t, dp[i, j - 1] + 1, dp[i - 1, j] + 1)

        if max(len1, len2) == 0:
            return 0

        return dp[len1][len2] / max(len1, len2)

    @staticmethod
    def calculate_jaro_winkler(output_ids):
        """
        计算jaro_winkler
        对于推荐结果多样性的度量，衡量Top-K结果中API序列两两之间的相似度，表示单个推荐结果中列表的多样性。ILS越大，代表推荐结果的多样性越好
        :param output_ids:
        :return:
        """
        # 输出序列个数
        seq_length = len(output_ids)
        if seq_length <= 1:
            return 0

        # 两两之间计算levenshtein距离
        jaro_winkler_list = []
        for i in range(seq_length):
            for j in range(i + 1, seq_length):
                jaro_winkler_list.append(Metric.jaro_winkler(output_ids[i], output_ids[j]))

        return (2 / (seq_length * (seq_length - 1))) * np.sum(jaro_winkler_list)

    @staticmethod
    def jaro(seq1, seq2):
        """
        jaro相似度度量
        :param seq1:
        :param seq2:
        :return:
        """
        # 保证sequence1的长度比sequence2的长度短
        if len(seq1) > len(seq2):
            s = seq1
            seq1 = seq2
            seq2 = s

        len1 = len(seq1)
        len2 = len(seq2)

        # 匹配窗口大小
        window = max(len1, len2) // 2 - 1

        # 匹配的字符数量
        m = 0

        # 字符转换的次数
        t = 0

        # sequence2匹配转换的个数
        seq2_matched = [False] * len2

        for i in range(len1):
            # 直接匹配
            if seq1[i] == seq2[i]:
                m += 1
                seq2_matched[i] = True
                continue

            # 换位匹配
            for j in range(window):
                j_index = i - j - 1
                if j_index >= 0 and not seq2_matched[j_index] and seq1[i] == seq2[j_index]:
                    seq2_matched[j_index] = True
                    m += 1
                    t += 1
                    break

            for j in range(window):
                j_index = i + j + 1
                if j_index < len2 and not seq2_matched[j_index] and seq1[i] == seq2[j_index]:
                    seq2_matched[j_index] = True
                    m += 1
                    t += 1
                    break

        if m == 0:
            return 0

        return 1 / 3 * (m / len1 + m / len2 + (m - t // 2) / m)

    @staticmethod
    def jaro_winkler(seq1, seq2):
        """
        计算jaro_winkler相似度
        :param seq1:
        :param seq2:
        :return:
        """
        jaro_score = Metric.jaro(seq1, seq2)
        l = 0
        for i in range(min(4, len(seq1), len(seq2))):
            if seq1[i] == seq2[i]:
                l += 1

        p = 0.1
        jaro_winkler_score = jaro_score + l * p * (1 - jaro_score)
        return 1 - jaro_winkler_score

    @staticmethod
    def calculate_coverage(count_dict):
        """
        计算覆盖率
        覆盖率：对于推荐结果多样性的度量，衡量推荐系统所推荐的API占所有API数的比例。覆盖率越高，代表推荐结果的多样性越好
        :param count_dict:
        :return:
        """
        return len(count_dict) / len(Metric.api_dataset)

    @staticmethod
    def calculate_tail_coverage(count_dict):
        """
        计算尾部覆盖率
        将最受欢迎的前20%项目表示为头部项目，其他项目是构成尾部项目集的尾部项目
        :param count_dict:
        :return:
        """
        tail_begin_index = Metric.api_dataset.iloc[0]['index'] + len(Metric.api_dataset) * 0.2
        tail_count = 0
        for index, apis in enumerate(count_dict):
            if apis[0] > tail_begin_index:
                tail_count += 1

        return tail_count / (len(Metric.api_dataset) * 0.8)


class DatasetTool:
    @staticmethod
    def collate_fn(batch):
        """
        取样本
        :param batch: 一个batch的数据
        :return:
        """
        # 得到一个batch中的所有数据
        question_list = [data["question"] for data in batch]
        api_description_list = [data["api_description"] for data in batch]
        api_sequence_list = [data["api_sequence"] for data in batch]

        # 对数据进行padding
        question_ids, question_mask = DatasetTool.padding(question_list)
        api_description_ids, api_description_mask = DatasetTool.padding_list(api_description_list)
        api_sequence_ids, _ = DatasetTool.padding(api_sequence_list)

        # 将label为1的部分置为-100
        api_sequence_ids[api_sequence_ids == 1] = -100

        return question_ids, question_mask, api_description_ids, api_description_mask, api_sequence_ids

    @staticmethod
    def padding(ids_list):
        """
        将list中的tensor padding到同样长度
        :param ids_list:
        :return:
        """
        max_length = max([len(ids) for ids in ids_list])
        ids_padding_list = []
        attention_padding_list = []
        for ids in ids_list:
            zero_tensor = torch.zeros(max_length - len(ids), dtype=torch.int64, device=device)
            one_tensor = torch.ones(max_length - len(ids), dtype=torch.int64, device=device)

            attention_tensor = torch.ones(len(ids), dtype=torch.int64, device=device)
            ids_padding_list.append(torch.cat((ids, one_tensor), dim=0))
            attention_padding_list.append(torch.cat((attention_tensor, zero_tensor), dim=0))
        return torch.stack(ids_padding_list), torch.stack(attention_padding_list)

    @staticmethod
    def padding_list(ids_list_list):
        """
        对list中含list的tensor进行padding
        :param ids_list_list:
        :return:
        """
        ids_padding_list = []
        attention_padding_list = []

        # 对batch中的每一条进行padding
        for ids in ids_list_list:
            api_desc_ids, api_desc_mask = DatasetTool.padding(ids)
            ids_padding_list.append(api_desc_ids)
            attention_padding_list.append(api_desc_mask)

        max_shape0 = max([ids.shape[0] for ids in ids_padding_list])
        max_shape1 = max([ids.shape[1] for ids in ids_padding_list])

        # padding 1维度
        for i in range(len(ids_padding_list)):
            zero_tensor = torch.zeros((ids_padding_list[i].shape[0], max_shape1 - ids_padding_list[i].shape[1]),
                                      dtype=torch.int64, device=device)
            one_tensor = torch.ones((ids_padding_list[i].shape[0], max_shape1 - ids_padding_list[i].shape[1]),
                                    dtype=torch.int64, device=device)

            ids_padding_list[i] = torch.cat((ids_padding_list[i], one_tensor), dim=1)
            attention_padding_list[i] = torch.cat((attention_padding_list[i], zero_tensor), dim=1)

        # padding 0维度
        for i in range(len(ids_padding_list)):
            if max_shape0 - ids_padding_list[i].shape[0] > 0:
                zero_tensor = torch.zeros((max_shape0 - ids_padding_list[i].shape[0], max_shape1), dtype=torch.int64,
                                          device=device)
                one_tensor = torch.ones((max_shape0 - ids_padding_list[i].shape[0], max_shape1), dtype=torch.int64,
                                        device=device)

                ids_padding_list[i] = torch.cat((ids_padding_list[i], one_tensor), dim=0)
                zero_tensor[:, 0] = 1
                attention_padding_list[i] = torch.cat((attention_padding_list[i], zero_tensor), dim=0)

        return torch.stack(ids_padding_list), torch.stack(attention_padding_list)

    @staticmethod
    def shift_right(input_ids):
        """
        对input id进行右移 使用1进行右移填充
        :param input_ids:
        :return:
        """
        # 确定开始符号和pad符号
        decoder_start_token_id = 0
        decoder_pad_token_id = -100

        # 对input_ids进行右移
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = decoder_start_token_id

        # 填充为-100的位置
        shifted_input_ids.masked_fill_(shifted_input_ids == decoder_pad_token_id, 1)

        return shifted_input_ids
