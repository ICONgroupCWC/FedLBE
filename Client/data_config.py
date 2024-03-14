import json
import yaml


async def get_data_config(job_data, websocket):
    data_set = job_data[0]

    data_config = yaml.safe_load(open('data/' + str(data_set) + '/config.yaml'))
    data_config['type'] = 'data'
    data_config = json.dumps(data_config)
    print('sending data config ' + str(data_config))
    await websocket.send(data_config)
