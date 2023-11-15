from DBService import db_connector
from DBService import db_util


def save_job_data(data, task_id):

    task_data = db_util.create_task_data(data, task_id)
    scheme_data = db_util.create_scheme_data(data, task_id)
    model_data = db_util.create_model_data(data, task_id)
    model_param = db_util.create_model_parameters(data, task_id)
    dataset_data = db_util.create_dataset_data(data, task_id)


    db_connector.insert('task', task_data)
    db_connector.insert('federated', scheme_data)
    db_connector.insert('model', model_data)
    db_connector.insert('model_parameters', model_param)
    db_connector.insert('dataset', dataset_data)

def save_results(data, task_id):

    results = db_util.create_results_data(data, task_id)
    db_connector.insert('results', results)



