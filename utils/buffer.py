import random
import copy
import numpy as np
from operator import itemgetter


class ReplayMemory:
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, ut, utn_hat_d, utn_hat_p, utn):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = tuple([copy.deepcopy(x) for x in [ut, utn_hat_d, utn_hat_p, utn]])
        #print(self.buffer[self.position])
        self.position = (self.position + 1) % self.capacity
        #print(self.position)

    def push_batch(self, batch):
        if len(self.buffer) < self.capacity:
            append_len = min(self.capacity - len(self.buffer), len(batch))
            self.buffer.extend([None] * append_len)

        if self.position + len(batch) < self.capacity:
            self.buffer[self.position : self.position + len(batch)] = batch
            self.position += len(batch)
        else:
            self.buffer[self.position : len(self.buffer)] = batch[:len(self.buffer) - self.position]
            self.buffer[:len(batch) - len(self.buffer) + self.position] = batch[len(self.buffer) - self.position:]
            self.position = len(batch) - len(self.buffer) + self.position

    def sample(self, batch_size):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        batch = random.sample(self.buffer, batch_size)
        ut, utn_hat_d, utn_hat_p, utn = map(np.stack, zip(*batch))
        return ut, utn_hat_d, utn_hat_p, utn
    
    def sample_near(self, batch_size):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        # batch = random.sample(self.buffer, batch_size)
        batch = self.buffer[-batch_size:]
        ut, utn_hat_d, utn_hat_p, utn = map(np.stack, zip(*batch))
        return ut, utn_hat_d, utn_hat_p, utn


    def sample_all_batch(self, batch_size):
        idxes = np.random.randint(0, len(self.buffer), batch_size)
        batch = list(itemgetter(*idxes)(self.buffer))
        ut, utn_hat_d, utn_hat_p, utn = map(np.stack, zip(*batch))
        return ut, utn_hat_d, utn_hat_p, utn
        
    def return_all(self):
        return self.buffer

    def __len__(self):
        return len(self.buffer)

# buffer = ReplayMemory(10, 0)

# for i in range(10):
#     ut = torch.rand((4,64,64))
#     utn_hat = ut*ut
#     utn = torch.rand((4,64,64))
#     buffer.push(ut, utn_hat, utn)
# for i in range(4):
#     ut, utn_hat, utn = buffer.sample(2+i)
#     print(ut.shape, utn_hat.shape)
    
# ut, utn_hat, utn = buffer.sample_all_batch(100)
# print(ut.shape, utn_hat.shape)
# ut, utn_hat, utn = buffer.sample_near(100)
# print(ut.shape, utn_hat.shape)
# ut, utn_hat, utn = buffer.sample(100)
# print(ut.shape, utn_hat.shape)
