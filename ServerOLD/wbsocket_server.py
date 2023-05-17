import asyncio
import logging
from concurrent.futures.process import ProcessPoolExecutor

import websockets
from Server.server_start_process import start_job
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
        logging.debug('producer exception catch ' +str(e))


async def listener(websocket, path):

    if path == '/job_receive':
        job_server = JobServer()
        async for message in websocket:
            print(str(message))
            job_data = json.loads(message)
            loop = asyncio.get_running_loop()
            async start_job(job_data,websocket)
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