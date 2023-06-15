# Federated Learning Backend

This repository contains the backend of the web solution  developed to facilitate federated learning which can be orchestrated using a web front end.Both client and server code can be found here

## Federated Learning

![architecture diagram](documents/architecture.png)

Code for client and server is provided in their respective folders. 

## Server
### dependencies

1. [pytorch](https://pytorch.org/) 2.0 or latest
2. [websockets](https://websockets.readthedocs.io/en/stable/intro/index.html#installation) (pip install websockets)
3. [torchvision](https://pypi.org/project/torchvision/)
4. [bson](https://pypi.org/project/bson/) (pip install bson)

Download Server code and run *wbsocket_server.py*. 

**The data to be tested against for calculating test accuracy during training should be stored in a subfolder in _data_ folder.** 













