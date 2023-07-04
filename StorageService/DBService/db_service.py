from . import connector


def get_model(user_name, task_name):
    print('get model ' + str(user_name) + ' ' + str(task_name))
    task_id = connector.get_task_id(user_name, task_name)

    results = connector.get_results('results', task_id)
    model = bytearray(results[0][-1])
    return model


def get_data(user_name, task_name):
    data = {}
    task_id = connector.get_task_id(user_name, task_name)
    task = connector.get_from_id('task', task_id)[0]
    federated = connector.get_from_id('federated', task_id)[0]
    model_parameters = connector.get_from_id('model_parameters', task_id)[0]
    dataset = connector.get_from_id('dataset', task_id)
    results = connector.get_from_id('results', task_id)
    print('results ' + str(results))

    # _, comm_rounds, train_loss, test_loss, round_time, test_accuracy,_ = results
    results = [(c, d, e,f) for a, b, c, d, e, f, g in results]
    print('results after ' + str(results))
    print('task ' + str(task))
    print('federated ' + str(federated))
    data['task'] = {'name': task[2], 'date': task[8], 'scheme': task[5], 'clients': len(task[7]),
                    'client_fraction': federated[4], 'comm_rounds': federated[6]}
    # data['task'] = task
    data['federated'] = {'name': task[2], 'minibatch_size': federated[1], 'local_epoch': federated[2],
                         'learning_rate': federated[3], 'test_batch_size': federated[5],
                         'optimizer': model_parameters[1], 'loss': model_parameters[2]}
    train_loss = []
    test_loss = []
    test_accuracy = []
    round_time = []

    for i in range(len(results)):
        train_loss.append(results[i][0])
        test_loss.append(results[i][1])
        round_time.append(results[i][2])
        test_accuracy.append(results[i][3])
    data['train_loss'] = train_loss
    data['test_loss'] = test_loss
    data['test_accuracy'] = test_accuracy
    data['round_time'] = round_time
    return data


def get_tasks(user_name):
    tasks = connector.get_all_tasks('task', user_name)
    tasks = tasks
    return tasks

#
# results = get_model('test', 'test17')
# # print(bytearray(results[0][-1]))
# model = bytearray(results[0][-1])
#
# f = open('model.pt', 'wb')
# f.write(model)
# f.close()
