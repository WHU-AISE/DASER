def config_data():
    conf = {
        'model_path': 'model',
        'train_path': 'data/train.feather',
        'test_path': 'data/test.feather',
        'api_path': 'data/api.feather',
        'trained_model_path': 'output/-1',
    }
    return conf


def config_model():
    conf = {
        'batch_size': 256,
        'epoch': 100,
        'random_seed': 42,
        'max_output_length': 32,
        'parallel': True,
        'gpu_ids': '0,1',
        'top_k': 10,
        'groups': 2,
        'lambda1': 0.5,
        'lambda2': 0.1,
    }
    return conf
