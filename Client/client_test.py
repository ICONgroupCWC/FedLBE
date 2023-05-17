import asyncio
import logging
import pickle
from concurrent.futures.process import ProcessPoolExecutor

import websockets
from client_process import process
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
    if path == '/process':

        async for message in websocket:
            print('received message')

            job_data = pickle.loads(message)
            print('pickle data ' + str(job_data))
            # job_data['client'] = 5000
            loop = asyncio.get_running_loop()
            await process(job_data, websocket)
            # loop.create_task(process(job_data, websocket))
            print('task done closing connection')
            await websocket.close()
            # job_server.start_job(job_data)


try:
    start_server = websockets.serve(listener, "0.0.0.0", 5000, ping_timeout=None, max_size=None)
    loop = asyncio.get_event_loop()

    loop.run_until_complete(start_server)
    loop.run_forever()
except Exception as e:
    print(f'Caught exception {e}')
    pass
finally:
    loop.close()
