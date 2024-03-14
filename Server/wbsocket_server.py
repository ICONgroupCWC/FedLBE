import asyncio
import logging
from concurrent.futures.process import ProcessPoolExecutor
import bson
import websockets
from server_start_process import JobServer
from processors.llm_processor import  LLMProcessor
from server_heterogenous_start import JobServerHetero
import json
from providers import llm_provider
from utils import create_dashboard_msg

task_executor = ProcessPoolExecutor(max_workers=3)


async def producer(websocket, message):
    log = logging.getLogger('producer')
    log.info('Received processed message')
    serialized_message = json.dumps(message)
    logging.debug('serial ' + str(serialized_message))
    try:
        await websocket.send(serialized_message)
    except Exception as e:
        logging.debug('producer exception catch ' + str(e))


async def listener(websocket, path):

    if path == '/job_receive':

        async for message in websocket:
            print('received message')
            print('message ' + str(message))
            job_data = bson.loads(message)
            print(job_data)
            # job_data = json.loads(ms['jobData'])
            local_loop = asyncio.get_running_loop()
            # await start_job(job_data, websocket)
            job_server = JobServer()
            local_loop.create_task(job_server.start_job(job_data, websocket))
            print('task created')
            # job_server.start_job(job_data)

    if path == '/job_receive_hetero':

        async for message in websocket:
            print('received message')
            print('message ' + str(message))
            job_data = bson.loads(message)
            print(job_data)
            # job_data = json.loads(ms['jobData'])
            local_loop = asyncio.get_running_loop()
            # await start_job(job_data, websocket)
            job_server = JobServerHetero()
            local_loop.create_task(job_server.start_job(job_data, websocket))
            print('task created')
            # job_server.start_job(job_data)

    if path == '/job_receive_llm':

        async for message in websocket:
            print('received message')
            print('message ' + str(message))
            data = bson.loads(message)
            print(data)
            user_prompt = data['jobData']['general']['prompt']

            job_data = llm_provider.get_job_config(user_prompt)
            dashboard_data = create_dashboard_msg(data, job_data)
            print('received job data ' + str(job_data))
            data['jobData']['scheme'] = job_data
            local_loop = asyncio.get_running_loop()
            await websocket.send(json.dumps(dashboard_data))
            processor = LLMProcessor()
            local_loop.create_task(processor.start_process(data, websocket))
            print('task created')



try:
    start_server = websockets.serve(listener, "0.0.0.0", 8200, ping_interval=None)
    loop = asyncio.get_event_loop()

    loop.run_until_complete(start_server)
    loop.run_forever()
except Exception as e:
    print(f'Caught exception {e}')
    pass
finally:
    loop.close()
