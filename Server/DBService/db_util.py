import uuid

schemes = {'FedL': 'Federated Learning', 'DistL': 'Distributed learning'}


def create_task_data(data, task_id):
    task_data = {}
    task_data['task_id'] = task_id
    task_data['user_name'] = 'test'
    task_data['task_name'] = data['general']['task']
    task_data['overview'] = data['general']['taskOverview']
    task_data['scheme_short'] = data['general']['method']
    task_data['scheme_long'] = schemes[str(data['general']['method'])]
    task_data['host'] = data['general']['host']
    task_data['clients'] = ", ".join(client['client_ip'] for client in data['general']['clients'])

    return task_data


def create_scheme_data(data, task_id):
    scheme_data = {}
    scheme = data['scheme']
    scheme_data['task_id'] = task_id
    scheme_data['minibatch'] = scheme['minibatch']
    scheme_data['local_epoch'] = scheme['epoch']
    scheme_data['learning_rate'] = scheme['lr']
    scheme_data['client_fraction'] = scheme['clientFraction']
    scheme_data['test_bs'] = scheme['minibatchtest']
    scheme_data['comm_rounds'] = scheme['comRounds']

    return scheme_data


def create_model_data(data, task_id):
    model_data = {}

    model_data['task_id'] = task_id
    model_data['description'] = data['modelData']['modelOverview']
    model_path = "./ModelData/" + str(task_id)

    with open(model_path + '/Model.py', 'rb') as f:
        filedata = f.read()
        model_data['model_arch'] = filedata

    return model_data


def create_model_parameters(data, task_id):
    model_param = {}

    model_param['task_id'] = task_id
    model_param['optimizer'] = data['modelParam']['optimizer']
    model_param['loss'] = data['modelParam']['loss']

    return model_param


def create_dataset_data(data, task_id):
    dataset_param = {}

    dataset_param['task_id'] = task_id
    dataset_param['data_type'] = data['preprocessing']['dtype']
    dataset_param['normalize'] = False if data['preprocessing']['normalize'] is None else data['preprocessing']['normalize']
    dataset_param['mean'] = 0 if (data['preprocessing']['normalize'] is None or not data['preprocessing']['normalize'] ) else data['preprocessing']['mean']
    dataset_param['std'] = 0 if (data['preprocessing']['normalize'] is None or not data['preprocessing']['normalize'] ) else data['preprocessing']['std']

    return dataset_param

def create_results_data(data, task_id):
    results = {}
    print(data)
    results['task_id'] = task_id
    results['comm_round'] = data['round']
    results['train_loss'] = data['train_loss']
    results['test_loss'] = data['test_loss']
    results['round_time'] = data['round_time']
    results['test_accuracy'] = data['accuracy']

    model_path = "./ModelData/" + str(task_id)

    with open(model_path + '/model.pt', 'rb') as f:
        filedata = f.read()
        results['model'] = filedata

    return results