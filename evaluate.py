import configs
import os

os.environ["CUDA_VISIBLE_DEVICES"] = getattr(configs, 'config_model')()['gpu_ids']

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from module import Metric, end_token
from tokenizer import tokenizer, api_tokenizer


def evaluate(model, data_loader, output_path=None, config_model=None, data_size=-1):
    """
    在测试集上对模型进行评估
    :param output_path: 输出测试的生成结果
    :param model: 训练好的模型
    :param data_loader: 数据集
    :return:
    """
    output_path = f"{output_path}/top{config_model['top_k']} - groups{config_model['groups']} - lambda1-{config_model['lambda1']} - lambda2-{config_model['lambda2']} - data{data_size}"

    if os.path.exists(output_path):
        print("File exists!")
        return

    # 写入文件
    os.makedirs(output_path, exist_ok=True)
    log_path = f"{output_path}/logs.log"
    output_id_path = f"{output_path}/output.feather"
    api_count_path = f"{output_path}/api_count.feather"
    log = open(log_path, "w", encoding="utf-8")

    # 保存数据信息
    output_id_list = []
    bleu_list = []
    meteor_list = []
    rouge_list = []
    levenshtein_distance_list = []
    jaro_winkler_list = []
    index = 0

    # 保存出现次数信息
    api_count_dict = {}

    for question_ids, question_mask, api_description_ids, api_description_mask, api_sequence_ids in tqdm(
            data_loader, position=0, leave=True):
        with torch.no_grad():
            output_ids = model.diverse_beam_search(input_ids=question_ids,
                                                   attention_mask=question_mask,
                                                   top_k=config_model['top_k'],
                                                   groups=config_model['groups'],
                                                   lambda1=config_model['lambda1'],
                                                   lambda2=config_model['lambda2'])

        # 对batch中的每一条数据进行遍历
        for output_index in range(len(output_ids)):
            # question数据
            question_id = question_ids[output_index]

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

            # 加入次数统计
            for id_list in output_id:
                for id in id_list:
                    api_count_dict[id] = api_count_dict.setdefault(id, 0) + 1

            # 计算bleu
            bleu = Metric.calculate_bleu(output_id, label_id)
            bleu_list.append(bleu)

            # 计算meteor
            meteor = Metric.calculate_meteor(output_id, label_id)
            meteor_list.append(meteor)

            # 计算rouge
            rouge = Metric.calculate_rouge(output_id, label_id)
            rouge_list.append(rouge)

            # 计算levenshtein_distance
            levenshtein_distance = Metric.calculate_levenshtein_distance(output_id)
            levenshtein_distance_list.append(levenshtein_distance)

            # 计算jaro_winkler
            jaro_winkler = Metric.calculate_jaro_winkler(output_id)
            jaro_winkler_list.append(jaro_winkler)

            # 将生成结果输出到文件
            if output_path is not None:
                log.write(f"Batch {index + 1}" + "\n")

                log.write(
                    "Question: \t" + str(tokenizer.decode(question_id).replace("</s>", "").replace("<pad>", "")) + "\n")

                log.write("Target: \t" + str(api_tokenizer.decode(label_id)) + "\n")

                output_text = api_tokenizer.batch_decode(output_id)
                for output_width_index, row in enumerate(output_text):
                    log.write(f"Output {output_width_index + 1}: \t" + str(row) + "\n")

                log.write(
                    f"BLEU: {round(bleu, 4)}, meteor: {round(meteor, 4)}, rouge: {round(rouge, 4)}, levenshtein distance: {round(levenshtein_distance, 4)}, jaro winkler: {round(jaro_winkler, 4)}")

                # 加入output_id的输出
                output_id_list.append(output_id)

            index += 1

    # 对计数字典进行排序
    api_count_dict = sorted(api_count_dict.items(), key=lambda d: d[0])

    # 保存输出id
    output_id_df = pd.DataFrame(columns=['output_id'])
    output_id_df['output_id'] = output_id_list
    output_id_df.to_feather(output_id_path)

    # 保存计数字典
    api_count_df = pd.DataFrame(columns=['api', 'count'])
    api_list = []
    count_list = []
    for row in api_count_dict:
        api_list.append(row[0])
        count_list.append(row[1])
    api_count_df['api'] = api_list
    api_count_df['count'] = count_list
    api_count_df.to_feather(api_count_path)

    # 计算指标
    coverage = Metric.calculate_coverage(api_count_dict)
    tail_coverage = Metric.calculate_tail_coverage(api_count_dict)

    # 写入总结果
    log.write(
        "------------------------------------------------------------------------------------------------------------------------\n")
    log.write(
        f"BLEU: {round(float(np.mean(bleu_list)), 4)}, meteor: {round(float(np.mean(meteor_list)), 4)}, rouge: {round(float(np.mean(rouge_list)), 4)}, levenshtein distance: {round(float(np.mean(levenshtein_distance_list)), 4)}, jaro winkler: {round(float(np.mean(jaro_winkler_list)), 4)}, coverage: {round(coverage, 4)}, tail_coverage: {round(tail_coverage, 4)}")
