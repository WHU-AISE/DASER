import pandas as pd
from transformers import RobertaTokenizer

import configs

config_data = getattr(configs, 'config_data')()

# 加载tokenizer
tokenizer = RobertaTokenizer.from_pretrained(config_data['model_path'])

# 加载api_tokenizer
api_tokenizer = RobertaTokenizer.from_pretrained(config_data['model_path'])
api_dataset = pd.read_feather(config_data['api_path'])
api_list = api_dataset['api'].tolist()
api_tokenizer.add_tokens(api_list)
