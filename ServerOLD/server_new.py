import asyncio
from concurrent.futures.process import ProcessPoolExecutor

import websockets
from server import JobServer
import json


task_executor = ProcessPoolExecutor(max_workers=3)
async def listener(websocket, path):

    if path == '/job_receive':
        job_server = JobServer()
        async for message in websocket:
            print(str(message))
            job_data = json.loads(message)
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(task_executor, job_server.start_job, job_data)
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