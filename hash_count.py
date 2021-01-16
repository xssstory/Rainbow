import numpy as np
import torch
import os

class HashTable(object):
  def __init__(self, args):
    self.table = {}
    self.state_action_table = {}
    self.state_action_count = 0
    self.hash_dim = args.hash_dim
    self.device = args.device
    self.state_dim = 46 * args.history_length if args.env_type == 'sepsis' else args.history_length * 6 if args.env_type == 'hiv' else 84 * 84 * args.history_length 
    self.A = torch.randn([self.hash_dim, self.state_dim], device=args.device)
      

  def step(self, state, action, count_action_state):
    hash_item, state_action_hash_item = self.hash_state(state, action)
    if hash_item in self.table:
      self.table[hash_item] += 1
    else:
      self.table[hash_item] = 1
    if count_action_state:
      if state_action_hash_item in self.state_action_table:
        self.state_action_table[state_action_hash_item] += 1
      else:
        self.state_action_table[state_action_hash_item] = 1
      self.state_action_count = self.state_action_table[state_action_hash_item]
    return self.table[hash_item]

  def hash_state(self, state, action):
    hidden = self.A.matmul(state.contiguous().view(-1, 1))
    state_hash = ''.join(map(lambda x: 'a' if x > 0 else 'b', hidden))
    state_action_hash = state_hash + str(action)
    return state_hash, state_action_hash

  def save(self, path, name='hash.pth'):
    torch.save({'A': self.A, 'table': self.table, 'state_action_table': self.state_action_table}, os.path.join(path, name))

  def load(self, path, name='hash.pth'):
    state = torch.load(os.path.join(path, name), map_location='cpu')
    self.A = state['A'].to(device=self.device)
    self.table = state['table']
    self.state_action_table = state.get('state_action_table', {})

