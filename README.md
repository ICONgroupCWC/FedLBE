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

**The data to be tested against for calculating test accuracy during training should be stored in a subfolder in _./data_ folder.** 

## Client

### dependencies

1. [pytorch](https://pytorch.org/) 2.0 or latest
2. [websockets](https://websockets.readthedocs.io/en/stable/intro/index.html#installation) (pip install websockets)
3. [torchvision](https://pypi.org/project/torchvision/)

run *python client_service.py 5000* (or any number for port) on the client device

**The data to train the model should be stored in a subfolder in _./data_ folder.** 

## Storage service

This section of the code is for managing a separate flask service for storing and downloading the trained model which will help in comparing the trained model on the front-end(user interface) 

### dependencies

1. [psycopg2](https://pypi.org/project/psycopg2/)
2. [flask](https://flask.palletsprojects.com/en/3.0.x/)
3. [pytorch](https://pytorch.org/) 2.0 or latest

Posdtgres database needed to be run in order for the storage service to run

Install postgres in your local machine by following the official [documentation](https://www.postgresql.org/download/).

Once postgres is installed create database and tables using the commands found in **feddb** text file.

Change port and db names accordingly in the file *connector.py*

Finally go to **StorageService** folder and run command *flask --app app run*

## Demo

Once all the services (Server, Client, Storage) are running FrontEnd need to be run to create a federated learning job using web Interface. (Instructions found on [FedLFrontEnd](https://github.com/ICONgroupCWC/FedLFrontEnd) repo) 

[Demo](https://youtu.be/f7e0BWT7t18?si=gOE1Nqp96N_lpOD9) for running the web app












