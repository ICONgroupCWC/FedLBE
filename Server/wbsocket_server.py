import asyncio
import logging
from concurrent.futures.process import ProcessPoolExecutor
import bson
import websockets
from Server.server_start_process import JobServer
import json

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
