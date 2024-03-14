import requests


def get_job_config(user_prompt):

    request_data = {'user_prompt': str(user_prompt)}
    r = requests.post('http://127.0.0.1:8000/get_config', json=request_data)
    configs = r.json()
    print(configs)
    return configs

def get_model_architecture(config):

    request_data = {"config": config}
    r = requests.post('http://127.0.0.1:8000/get_model_architecture', json=request_data)

    print(r.text)
    return r.text

