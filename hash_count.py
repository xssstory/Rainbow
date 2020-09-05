import numpy as np
import torch
import os

class HashTable(object):
  def __init__(self, args):
    self.table = {}
    self.hash_dim = args.hash_dim
    self.device = args.device
    self.state_dim = 46 * args.history_length if args.env_type == 'sepsis' else args.history_length * 6 if args.env_type == 'hiv' else 84 * 84 * args.history_length 
    self.A = torch.randn([self.hash_dim, self.state_dim], device=args.device)
      

  def step(self, state, action):
    hash_item = self.hash_state(state, action)
    if hash_item in self.table:
      self.table[hash_item] += 1
    else:
      self.table[hash_item] = 1
    return self.table[hash_item]

  def hash_state(self, state, action):
    hidden = self.A.matmul(state.contiguous().view(-1, 1))
    return ''.join(map(lambda x: 'a' if x > 0 else 'b', hidden))

  def save(self, path, name='hash.pth'):
    torch.save({'A': self.A, 'table': self.table}, os.path.join(path, name))

  def load(self, path, name='hash.pth'):
    state = torch.load(os.path.join(path, name), map_location='cpu')
    self.A = state['A'].to(device=self.device)
    self.table = state['table']

