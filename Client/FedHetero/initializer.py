import json
import os
import pickle


async def initialize(job_data, websocket):

    ext_file = job_data[0]
    rep_file = job_data[1]
    job_id = job_data[2]

    rep_learner_file = "./ModelData/" + str(job_id) + '/RepModel.py'
    extractor_file = "./ModelData/" + str(job_id) + '/ExtModel.py'

    os.makedirs(os.path.dirname(rep_learner_file), exist_ok=True)
    os.makedirs(os.path.dirname(extractor_file), exist_ok=True)

    with open(rep_learner_file, 'wb') as f:
        f.write(rep_file)

    with open(extractor_file, 'wb') as f:
        f.write(ext_file)

    message = pickle.dumps({'status': 'done', 'message': 'saved model'})

    await websocket.send(message)



