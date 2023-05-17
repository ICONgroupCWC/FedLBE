from torchvision import datasets, transforms
import pickle
from multiprocessing import Process, Queue, Value, Event
import zmq.asyncio
import zmq
from Client.client_update import ClientUpdate
from Model import MLP_Net




def load_dataset():
    transforms_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    mnist_data_train = datasets.MNIST('./data/mnist', train=True, download=True, transform=transforms_mnist)
    mnist_data_test = datasets.MNIST('./data/mnist', train=False, download=True, transform=transforms_mnist)

    return mnist_data_train, mnist_data_test


def receive_message(q,):
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5555")
    progress = Value('i', -1)
    max_epochs = Value('i', 0)
    weights = Queue()
    loss = Queue()
    training = False
    done_event = Event()
    while True:

        if progress.value == max_epochs.value:
            print('training stopped')
            training = False
            progress.value = -1
            max_epochs.value = 0
        message = socket.recv()

        if not training and message != b'results':
            done_event.clear()
            message = pickle.loads(message)
            q.put(message)
        else:
            message = str(message,'utf-8')

        if message[0] == 'init':
            p2 = Process(target=process_msg, args=(q, progress, max_epochs, weights, loss, done_event))
            p2.start()
            training = True
            out_message = 'processing ' + str(progress.value)
            socket.send(out_message.encode())

        elif message == 'results':

            result = [weights.get(), loss.get()]
            out_message = pickle.dumps(result)
            socket.send(out_message)
            done_event.set()
        else:
            out_message = 'processing ' + str(progress.value)
            socket.send(out_message.encode())





def process_msg(q, progress,max_epochs, weights, loss, done_event):
    print('Starting process')
    message = q.get()
    task = message[0]

    if task == 'init':
        ds, test = load_dataset()
        print('ds type ' + str(type(ds)))
        model = MLP_Net()
        recvd = message
        # print(recvd[0])
        B = recvd[1]
        eta = recvd[2]
        E = recvd[3]
        ids = recvd[4]
        global_weights = recvd[5]
        model.load_state_dict(global_weights)
        max_epochs.value = E
        client = ClientUpdate(dataset=ds, batchSize=B, learning_rate=eta, epochs=E, idxs=ids)
        w , l = client.train(model, progress)
        weights.put(w)
        loss.put(l)
        print('extracted weight and loss')
        done_event.wait()


if __name__ == '__main__':

    q = Queue()
    p1 = Process(target=receive_message, args=(q,))

    p1.start()
