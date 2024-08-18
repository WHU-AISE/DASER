# DASER
DASER: Diversified API SEquence Recommendation based on Transformer and Diverse API Beam Search
## Project Structure

```Project Structure
.
│  configs.py
│  datasets.py
│  evaluate.py
│  model.py
│  module.py
│  README.md
│  test.py
│  tokenizer.py
│  train.py
│
├─data
│  │  api.feather
│  │  test.feather
│  │  train.feather
│  │
│  ├─java
│  │      api.feather
│  │      test.feather
│  │      train.feather
│  │
│  └─python
│          api.feather
│          test.feather
│          train.feather
│
└─model
   │  added_tokens.json
   │  config.json
   │  merges.txt
   │  pytorch_model.bin
   │  special_tokens_map.json
   │  tokenizer_config.json
   │  vocab.json
   │
   ├─java
   │      config.json
   │      pytorch_model.bin
   │
   └─python
          config.json
          pytorch_model.bin
```
