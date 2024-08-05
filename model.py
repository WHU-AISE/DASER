import pandas as pd
import torch
from torch.nn import CrossEntropyLoss
from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput

import configs
from module import DatasetTool, begin_token, end_token


class Model(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)

        self.config_data = getattr(configs, 'config_data')()
        self.config_model = getattr(configs, 'config_model')()

        self.api_dataset = pd.read_feather(self.config_data['api_path'])
        self.api_description = self.api_dataset.set_index('index')['tokenized_description'].to_dict()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            api_description_ids=None,
            api_description_mask=None,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                decoder_head_mask = head_mask

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = DatasetTool.shift_right(labels)

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # 训练时使用description的embedding
        if api_description_ids is not None:
            # 创建为0的start tensor
            start_tensor = torch.zeros(1, dtype=torch.int64)
            start_attention_tensor = torch.ones(1, dtype=torch.int64)

            # 对start tensor进行padding
            start_padding_tensor = torch.cat(
                (start_tensor, torch.ones(api_description_ids.shape[2] - start_tensor.shape[0], dtype=torch.int64)),
                dim=0).unsqueeze(0).unsqueeze(0)

            start_attention_padding_tensor = torch.cat((start_attention_tensor, torch.zeros(
                api_description_ids.shape[2] - start_attention_tensor.shape[0], dtype=torch.int64)),
                                                       dim=0).unsqueeze(0).unsqueeze(0)

            # 将其扩展维度
            start_padding_tensor = start_padding_tensor.expand(api_description_ids.shape[0], 1,
                                                               api_description_ids.shape[2]).to(self.device)
            start_attention_padding_tensor = start_attention_padding_tensor.expand(
                api_description_ids.shape[0], 1, api_description_ids.shape[2]).to(self.device)

            # 对description进行右移(shift_right)
            api_description_ids = torch.cat((start_padding_tensor, api_description_ids), dim=1)[:, :-1, :]
            api_description_mask = torch.cat((start_attention_padding_tensor, api_description_mask), dim=1)[:, :-1,
                                   :]

            # 得到api描述的embedding
            api_description_ids_embedding = self.encoder.get_input_embeddings()(api_description_ids)

            # 计算每个api描述的长度
            api_description_ids_embedding_length = api_description_mask.sum(dim=2).unsqueeze(-1).expand(
                api_description_mask.shape[0], api_description_mask.shape[1],
                api_description_mask.shape[2]).unsqueeze(
                -1).expand(api_description_mask.shape[0], api_description_mask.shape[1],
                           api_description_mask.shape[2],
                           self.config.d_model)

            # api描述的embedding的mask
            api_description_ids_embedding_mask = api_description_mask.unsqueeze(-1).expand(
                api_description_mask.shape[0], api_description_mask.shape[1], api_description_mask.shape[2],
                self.config.d_model)

            # 得到经过mask的embedding
            api_description_ids_embedding_masked = api_description_ids_embedding * api_description_ids_embedding_mask

            # 得到api序列的句表示
            api_description_output = (
                    api_description_ids_embedding_masked / api_description_ids_embedding_length).sum(
                dim=2)

            # 得到原label的embedding编码
            inputs_embeds = self.encoder.get_input_embeddings()(decoder_input_ids)

            # 将两个编码相加
            decoder_inputs_embeds = api_description_output + inputs_embeds
            decoder_input_ids = None

        # 生成时创建decoder_inputs_embeds
        elif decoder_input_ids is not None:
            decoder_inputs_embeds_list = []

            # 对batch中的每条数据都进行创建
            for batch_index in range(decoder_input_ids.shape[0]):
                batch_decoder_input_ids = decoder_input_ids[batch_index, :]

                batch_decoder_embeds_list = []

                # 对input_id进行逐个编码
                for id_index in range(batch_decoder_input_ids.shape[0]):
                    # 得到当前api
                    d_input_ids = batch_decoder_input_ids[id_index]

                    # 得到api的描述并编码
                    if d_input_ids.item() in self.api_description:
                        input_ids_api_description = self.api_description[d_input_ids.item()]
                    else:
                        input_ids_api_description = [d_input_ids.item()]
                    input_ids_api_description_tensor = torch.as_tensor(input_ids_api_description, dtype=torch.int64,
                                                                       device=self.device)
                    input_ids_api_embedding = self.encoder.get_input_embeddings()(input_ids_api_description_tensor)
                    batch_decoder_embeds_list.append(input_ids_api_embedding.mean(dim=0))

                # stack每个api的编码
                batch_decoder_inputs_outputs = torch.stack(batch_decoder_embeds_list)

                # 得到原label的embedding编码
                batch_inputs_embeds = self.encoder.get_input_embeddings()(batch_decoder_input_ids)

                # 将api表示和api描述相加
                batch_decoder_inputs_embeds = batch_decoder_inputs_outputs + batch_inputs_embeds
                decoder_inputs_embeds_list.append(batch_decoder_inputs_embeds)

            # 汇总一个batch内的embedding
            decoder_inputs_embeds = torch.stack(decoder_inputs_embeds_list)
            decoder_input_ids = None

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def diverse_beam_search(self, input_ids=None, attention_mask=None, top_k=1, groups=1, lambda1=1,
                            lambda2=1):
        """
        多样化beam_search
        :param input_ids: 输入id
        :param attention_mask: 输入注意力掩码
        :param top_k: 生成个数
        :param groups: 生成组数
        :param lambda1: 参数1 控制序列多样性
        :param lambda2: 参数2 控制流行度多样性
        :return:
        """
        # 流行度偏差
        diverse_tensor = torch.tensor(self.api_dataset['idf'], dtype=torch.float32, device=self.device)
        diverse_tensor = diverse_tensor * lambda2
        diverse_tensor = torch.cat(
            (torch.zeros(self.config.vocab_size - len(self.api_dataset), device=self.device), diverse_tensor), dim=0)

        batch_group_output_list = []
        for batch_index in range(input_ids.shape[0]):
            # 输入id与mask
            batch_input_ids = torch.as_tensor(input_ids[batch_index, :], device=self.device,
                                              dtype=torch.int64).unsqueeze(0)
            batch_attention_mask = torch.as_tensor(attention_mask[batch_index, :], device=self.device,
                                                   dtype=torch.int64).unsqueeze(0)

            # 初始化每个分组的保存列表
            output_ids_list = []
            group_size_list = []
            needed_list = []
            output_ids_set = set()

            # 初始化每个列表内内容
            for group_index in range(groups):
                # 计算每个分组最后所应有的api序列数量
                group_size = int(top_k / groups)
                if top_k % groups > group_index:
                    group_size += 1
                group_size_list.append(group_size)

            # 计算每个分组应该生成的api序列数量
            needed_list.append(group_size_list[0])
            for i in range(1, groups):
                needed_list.append(needed_list[i - 1] + group_size_list[i])

            with torch.no_grad():
                for group_index in range(groups):
                    # 取出该组所需的数据
                    input_decoder_ids = torch.as_tensor(begin_token, device=self.device, dtype=torch.int64).view(1, -1)
                    beam_score = torch.zeros(1, dtype=torch.int64, device=self.device)
                    output_ids = []

                    # 重复惩罚
                    repeat_punish_tensor = torch.zeros((1, self.config.vocab_size), dtype=torch.float32,
                                                       device=self.device)

                    for step in range(self.config_model['max_output_length']):
                        # 更新当前时间步的重复惩罚
                        if group_index != 0:
                            for output in output_ids_list:
                                if step + 1 < len(output) and output[step + 1] != end_token:
                                    repeat_punish_tensor[:, output[step + 1]] += 1

                        # 得到序列中该时间步每个节点的得分
                        step_input_ids = batch_input_ids.view(1, -1).repeat(input_decoder_ids.shape[0], 1)
                        step_attention_mask = batch_attention_mask.view(1, -1).repeat(input_decoder_ids.shape[0],
                                                                                      1)

                        scores = self.forward(input_ids=step_input_ids, attention_mask=step_attention_mask,
                                              decoder_input_ids=input_decoder_ids)[0]

                        # 生成第一个节点时需要将输入扩展到top_k的维度 进行之后的beam_search
                        if step == 0:
                            input_decoder_ids = input_decoder_ids.view(1, -1).repeat(
                                needed_list[group_index], 1)

                        scores = scores[:, -1]

                        # 得到原始得分
                        logit_score = torch.log_softmax(scores, dim=-1)
                        logit_score[:, :end_token] = -float('Inf')
                        logit_score[:, end_token + 1:self.config.vocab_size - len(self.api_dataset)] = -float('Inf')

                        # 向除了初始组之外的所有组内加入重复惩罚
                        if group_index != 0:
                            # 加入流行度惩罚
                            logit_score = torch.add(logit_score, diverse_tensor * group_index)

                            # 加入重复惩罚
                            logit_score = torch.add(logit_score,
                                                    -repeat_punish_tensor.repeat(scores.shape[0], 1) * lambda1)

                        # 得到序列概率得分
                        logit_score = beam_score.view(-1, 1) + logit_score

                        # 展平
                        logit_score = logit_score.view(-1)

                        # 获取下一个生成的api
                        next_score, next_token_id = torch.topk(logit_score, needed_list[group_index] - len(output_ids))

                        # 行索引
                        row_index = (next_token_id // scores.shape[-1])

                        # 列索引
                        column_index = (next_token_id % scores.shape[-1]).long().reshape(-1, 1)

                        # 更新得分
                        beam_score = next_score

                        # 连接api序列与当前时间步生成的api 作为模型的新的输入
                        input_decoder_ids = torch.cat([input_decoder_ids[row_index], column_index],
                                                      dim=1).long()

                        # 计算已完成的索引
                        flag = (column_index != end_token).view(-1)

                        # 只保留未搜索到终点的序列
                        beam_score = beam_score[flag]

                        # 加入到已完成节点中
                        output_ids += input_decoder_ids[flag == False].tolist()
                        input_decoder_ids = input_decoder_ids[flag]

                        # 完成搜索
                        if len(output_ids) >= needed_list[group_index]:
                            if group_index == 0:
                                output_ids_list = output_ids_list + output_ids[:needed_list[group_index]]
                                for output in output_ids_list:
                                    output_ids_set.add(str(output))
                            else:
                                # 计算该组所需的序列数
                                group_size = group_size_list[group_index]

                                # 加入不重复的序列 直至完成所需的序列数
                                output_index = 0
                                while group_size != 0 and output_index < len(output_ids):
                                    if str(output_ids[output_index]) not in output_ids_set:
                                        output_ids_list.append(output_ids[output_index])
                                        output_ids_set.add(str(output_ids[output_index]))
                                        group_size -= 1
                                    output_index += 1
                            break
            batch_group_output_list.append(output_ids_list)

        return batch_group_output_list
