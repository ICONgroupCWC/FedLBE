import numpy as np


class Scheduler:

    def __init__(self, scheduler_type, no_of_clients, client_fraction=0.0, avg_rounds=1):

        self.scheduler = scheduler_type
        self.no_of_clients = no_of_clients
        self.client_fraction = client_fraction
        self.avg_rounds = avg_rounds
        if scheduler_type == 'round_robin':
            self.__rr_list = np.arange(self.no_of_clients)
            # np.random.shuffle(self.__rr_list)
        elif scheduler_type == 'latency':
            self.latencies = np.ones((self.avg_rounds, self.no_of_clients), dtype='float')

    def get_workers(self, new_latencies=None):

        if self.scheduler == 'random':

            m = max(int(self.client_fraction * self.no_of_clients), 1)

            return np.random.choice(range(self.no_of_clients), m, replace=False)

        elif self.scheduler == 'round_robin':
            m = max(int(self.client_fraction * self.no_of_clients), 1)
            scheduled_idx = self.__rr_list[np.arange(0, m)]
            self.__rr_list = np.delete(self.__rr_list, np.arange(0, m))
            self.__rr_list = np.append(self.__rr_list, scheduled_idx)

            return scheduled_idx

        elif self.scheduler == 'full':
            return np.arange(0, self.no_of_clients)

        elif self.scheduler == 'latency':
            m = max(int(self.client_fraction * self.no_of_clients), 1)
            if new_latencies is None:
                new_latencies = np.ones((1, self.no_of_clients)) * 0.01
            self.latencies = np.append(self.latencies, new_latencies, axis=0)
            self.latencies = np.delete(self.latencies, 0, axis=0)
            print('self latencies ' + str(self.latencies))
            inv_latencies = 1 / self.latencies
            p_aux = inv_latencies.mean(axis=0) / sum(inv_latencies.mean(axis=0))
            print(p_aux)
            scheduled_idx = np.random.choice(range(self.no_of_clients), m, replace=False, p=p_aux)
            return scheduled_idx


# clients = [5000, 5001, 5002, 5003, 5004, 5005, 5006]
# scheduler = Scheduler('full', len(clients), 0.7)
# for i in range(10):
#     S_t = scheduler.get_workers()
#     print(S_t)
