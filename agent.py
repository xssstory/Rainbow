# -*- coding: utf-8 -*-
from __future__ import division
import os
import numpy as np
import torch
from torch import optim
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
import copy
import math
from collections import deque
import time

from model import DQN, SepsisDqn
DQN_DIC = {
  'atari': DQN,
  'sepsis': SepsisDqn,
  'hiv': SepsisDqn,
}

class Agent():
  def __init__(self, args, env):
    self.dqn_model = DQN_DIC[args.env_type]
    self.action_space = env.action_space()
    self.atoms = args.atoms
    self.Vmin = args.V_min
    self.Vmax = args.V_max
    self.support = torch.linspace(args.V_min, args.V_max, self.atoms).to(device=args.device)  # Support (range) of z
    self.delta_z = (args.V_max - args.V_min) / (self.atoms - 1)
    self.batch_size = args.batch_size
    self.n = args.multi_step
    self.discount = args.discount
    self.norm_clip = args.norm_clip
    self.num_deploy = 0
    self.deploy_policy = args.deploy_policy
    if self.deploy_policy == 'exp':
      self.exponent = 0
      self.exp_base = args.exp_base
    if self.deploy_policy == "policy_diverge":
      self.ratio = deque(maxlen=100)
    if self.deploy_policy == "policy" or self.deploy_policy == "reset_policy" or args.record_action_diff:
      self.action_diff = deque(maxlen=100)
    if self.deploy_policy in ['dqn-feature', 'reset_feature', 'dqn-feature-min', 'reset_feature_force', 'feature_lowf'] or args.record_feature_sim:
      self.feature_sim = deque(maxlen=100)
    if self.deploy_policy and self.deploy_policy.endswith('-min'):
      if isinstance(args.min_interval, int):
        self.min_interval = args.min_interval
        self.adapt_min_interval = False
      else:
        self.adapt_min_interval = True
        self.min_interval = int(args.min_interval.split('.')[-1])
        self.min_interval_update = self.min_interval
      self.cur_interval = 0
    self.last_update_T = 0

    self.deploy_net = self.dqn_model(args, self.action_space).to(device=args.device)
    if args.model:  # Load pretrained model if provided
      if os.path.isfile(args.model):
        state_dict = torch.load(args.model, map_location='cpu')  # Always load tensors onto CPU by default, will shift to GPU if necessary
        if 'conv1.weight' in state_dict.keys():
          for old_key, new_key in (('conv1.weight', 'convs.0.weight'), ('conv1.bias', 'convs.0.bias'), ('conv2.weight', 'convs.2.weight'), ('conv2.bias', 'convs.2.bias'), ('conv3.weight', 'convs.4.weight'), ('conv3.bias', 'convs.4.bias')):
            state_dict[new_key] = state_dict[old_key]  # Re-map state dict for old pretrained models
            del state_dict[old_key]  # Delete old keys for strict load_state_dict
        self.deploy_net.load_state_dict(state_dict)
        print("Loading pretrained model: " + args.model)
      else:  # Raise error if incorrect model path provided
        raise FileNotFoundError(args.model)
    if self.deploy_policy is None:
      self.online_net = self.deploy_net
      assert self.online_net is self.deploy_net
    else:
      self.online_net = copy.deepcopy(self.deploy_net)
      for param in self.deploy_net.parameters():
        param.requires_grad = False

    self.online_net.train()
    self.deploy_net.train()

    self.target_net = self.dqn_model(args, self.action_space).to(device=args.device)
    self.update_target_net()
    self.target_net.train()
    for param in self.target_net.parameters():
      param.requires_grad = False

    self.optimiser = optim.Adam(self.online_net.parameters(), lr=args.learning_rate, eps=args.adam_eps)

  # Resets noisy weights in all linear layers (of online net only)
  def reset_noise(self):
    self.online_net.reset_noise()
    # if self.deploy_policy is not None:
    #   self.deploy_net.reset_noise()

  # Acts based on single state (no batch)
  def act(self, state):
    with torch.no_grad():
      return (self.deploy_net(state.unsqueeze(0)) * self.support).sum(2).argmax(1).item()

  # Acts with an ε-greedy policy (used for evaluation only)
  def act_e_greedy(self, state, epsilon=0.001):  # High ε can reduce evaluation scores drastically
    return np.random.randint(0, self.action_space) if np.random.random() < epsilon else self.act(state)

  def learn(self, mem):
    # Sample transitions
    idxs, states, actions, returns, next_states, nonterminals, weights = mem.sample(self.batch_size)

    # Calculate current state probabilities (online network noise already sampled)
    log_ps = self.online_net(states, log=True)  # Log probabilities log p(s_t, ·; θonline)
    log_ps_a = log_ps[range(self.batch_size), actions]  # log p(s_t, a_t; θonline)

    with torch.no_grad():
      # Calculate nth next state probabilities
      pns = self.online_net(next_states)  # Probabilities p(s_t+n, ·; θonline)
      dns = self.support.expand_as(pns) * pns  # Distribution d_t+n = (z, p(s_t+n, ·; θonline))
      argmax_indices_ns = dns.sum(2).argmax(1)  # Perform argmax action selection using online network: argmax_a[(z, p(s_t+n, a; θonline))]
      self.target_net.reset_noise()  # Sample new target net noise
      pns = self.target_net(next_states)  # Probabilities p(s_t+n, ·; θtarget)
      pns_a = pns[range(self.batch_size), argmax_indices_ns]  # Double-Q probabilities p(s_t+n, argmax_a[(z, p(s_t+n, a; θonline))]; θtarget)

      # Compute Tz (Bellman operator T applied to z)
      Tz = returns.unsqueeze(1) + nonterminals * (self.discount ** self.n) * self.support.unsqueeze(0)  # Tz = R^n + (γ^n)z (accounting for terminal states)
      Tz = Tz.clamp(min=self.Vmin, max=self.Vmax)  # Clamp between supported values
      # Compute L2 projection of Tz onto fixed support z
      b = (Tz - self.Vmin) / self.delta_z  # b = (Tz - Vmin) / Δz
      l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)
      # Fix disappearing probability mass when l = b = u (b is int)
      l[(u > 0) * (l == u)] -= 1
      u[(l < (self.atoms - 1)) * (l == u)] += 1

      # Distribute probability of Tz
      m = states.new_zeros(self.batch_size, self.atoms)
      offset = torch.linspace(0, ((self.batch_size - 1) * self.atoms), self.batch_size).unsqueeze(1).expand(self.batch_size, self.atoms).to(actions)
      m.view(-1).index_add_(0, (l + offset).view(-1), (pns_a * (u.float() - b)).view(-1))  # m_l = m_l + p(s_t+n, a*)(u - b)
      m.view(-1).index_add_(0, (u + offset).view(-1), (pns_a * (b - l.float())).view(-1))  # m_u = m_u + p(s_t+n, a*)(b - l)

    loss = -torch.sum(m * log_ps_a, 1)  # Cross-entropy loss (minimises DKL(m||p(s_t, a_t)))
    self.online_net.zero_grad()
    (weights * loss).mean().backward()  # Backpropagate importance-weighted minibatch loss
    clip_grad_norm_(self.online_net.parameters(), self.norm_clip)  # Clip gradients by L2 norm
    self.optimiser.step()

    mem.update_priorities(idxs, loss.detach().cpu().numpy())  # Update priorities of sampled transitions

  def update_target_net(self):
    self.target_net.load_state_dict(self.online_net.state_dict())
  
  def update_deploy_net(self, T, args, mem, is_reset=False):
    if self.deploy_policy is None:
      # self.deploy_net.load_state_dict(self.online_net.state_dict())
      assert self.deploy_net is self.online_net
      self.num_deploy += 1
    elif self.deploy_policy == 'fixed' and T % args.delploy_interval == 0:
      assert self.deploy_net is not self.online_net
      self.deploy_net.load_state_dict(self.online_net.state_dict())
      self.num_deploy += 1
      with torch.no_grad():
        if hasattr(self, "action_diff") or hasattr(self, "feature_sim"):
          idxs, states, actions, returns, next_states, nonterminals, weights = mem.sample_recent(args.switch_bsz)
        if hasattr(self, "action_diff"):
          deploy_action = (self.deploy_net(states) * self.support).sum(2).argmax(1)
          online_action = (self.online_net(states) * self.support).sum(2).argmax(1)
          diff = 1 - deploy_action.eq(online_action).sum().item() / args.switch_bsz
          self.action_diff.append(diff)
        if hasattr(self, "feature_sim"):
          deploy_feature2 = self.deploy_net.extract(states).detach()
          online_feature2 = self.online_net.extract(states).detach()
          deploy_feature2 = F.normalize(deploy_feature2)
          online_feature2 = F.normalize(online_feature2)
          sim2 = deploy_feature2.mm(online_feature2.T)
          sim = sim2.diagonal().mean()
          self.feature_sim.append(sim.item())
    elif self.deploy_policy == 'exp':
      if (T - args.learn_start // args.replay_frequency) >= self.exp_base ** self.exponent:
        assert self.deploy_net is not self.online_net
        self.deploy_net.load_state_dict(self.online_net.state_dict())
        self.num_deploy += 1
        self.exponent += 1
    elif self.deploy_policy == "reset":
      if is_reset:
        self.deploy_net.load_state_dict(self.online_net.state_dict())
        self.num_deploy += 1
    else:
      # if T % args.delploy_interval != 0:
      #   return
      # start_time = time.time()
      if args.switch_memory_priority:
        idxs, states, actions, returns, next_states, nonterminals, weights = mem.sample2(args.switch_bsz)
      else:
        if args.switch_sample_strategy == "recent":
          idxs, states, actions, returns, next_states, nonterminals, weights = mem.sample_recent(args.switch_bsz)
        elif args.switch_sample_strategy == "uniform":
          idxs, states, actions, returns, next_states, nonterminals, weights = mem.uniform_sample_from_recent(args.switch_bsz, args.switch_memory_capcacity)
        else:
          raise RuntimeError("switch_sample_strategy {} is not supported !".format(args.switch_sample_strategy))
      # sample_time = time.time() - start_time
      # print("sample time", sample_time)

      if self.deploy_policy in ['dqn-feature', 'reset_feature', 'dqn-feature-min', 'feature_lowf']:
        if self.deploy_policy == "dqn-feature-min":
          if self.cur_interval < self.min_interval:
            self.cur_interval += 1
            return
        self.eval()
        with torch.no_grad():
          #deploy_feature = self.deploy_net.extract(states).detach().cpu().numpy()
          #online_feature = self.online_net.extract(states).detach().cpu().numpy()
          deploy_feature2 = self.deploy_net.extract(states).detach()
          online_feature2 = self.online_net.extract(states).detach()
          deploy_feature2 = F.normalize(deploy_feature2)
          online_feature2 = F.normalize(online_feature2)
          sim2 = deploy_feature2.mm(online_feature2.T)
          sim = sim2.diagonal().mean()
          if hasattr(self, "feature_sim"):
            self.feature_sim.append(sim.item())
        #sim = np.dot(deploy_feature, online_feature.T) \
        #/(np.linalg.norm(deploy_feature, axis=1, keepdims=True)* np.linalg.norm(online_feature, axis=1, keepdims=True))
        #sim = sim.diagonal().mean()
        #print(sim, sim2)
        self.train()
        if sim < args.feature_threshold:
          self.deploy_net.load_state_dict(self.online_net.state_dict())
          self.num_deploy += 1
          if self.deploy_policy == 'dqn-feature-min':
            self.cur_interval = 1
            if self.adapt_min_interval and self.min_interval < 10000:
              self.min_interval += self.min_interval_update
      elif self.deploy_policy == "reset_feature_force":
        need_deploy = False
        if T - self.last_update_T > 1000:
          need_deploy = True
        elif is_reset:
          self.eval()
          with torch.no_grad():
            deploy_feature2 = self.deploy_net.extract(states).detach()
            online_feature2 = self.online_net.extract(states).detach()
            deploy_feature2 = F.normalize(deploy_feature2)
            online_feature2 = F.normalize(online_feature2)
            sim2 = deploy_feature2.mm(online_feature2.T)
            sim = sim2.diagonal().mean()
            if hasattr(self, "feature_sim"):
              self.feature_sim.append(sim.item())
          self.train()
          if sim < args.feature_threshold:
            need_deploy = True
        if need_deploy:
          self.deploy_net.load_state_dict(self.online_net.state_dict())
          self.num_deploy += 1
          self.last_update_T = T 
      elif self.deploy_policy == 'q-value':
        self.eval()
        with torch.no_grad():
          deploy_value = (self.deploy_net(states) * self.support).sum(2)[range(args.switch_bsz), actions]
          online_value = (self.online_net(states) * self.support).sum(2)[range(args.switch_bsz), actions]
          if (abs(deploy_value - online_value) / deploy_value.masked_fill(deploy_value==0, 1)).mean() > args.q_value_threshold:
            self.deploy_net.load_state_dict(self.online_net.state_dict())
            self.num_deploy += 1
        self.train()
      elif self.deploy_policy == 'policy_diverge':
        self.eval()
        with torch.no_grad():
          # start_time = time.time()
          deploy_q = (self.deploy_net(states) * self.support).sum(2)  # (batch_size, action_dim)
          online_q = (self.online_net(states) * self.support).sum(2)
          # forward_time = time.time() - start_time
          # print("forward time", forward_time)
          deploy_dist = torch.distributions.categorical.Categorical(probs=torch.exp(deploy_q))
          online_dist = torch.distributions.categorical.Categorical(probs=torch.exp(online_q))
          new_action = online_q.argmax(1)  # (batch_size,)
          # p(action|new policy) / p(action|deploy policy)
          logp = online_dist.log_prob(new_action)  # (batch_size,)
          old_logp = deploy_dist.log_prob(new_action)
          ratio = torch.exp(logp - old_logp)
          self.ratio.append(ratio.mean().item())
          if torch.abs(ratio.mean() - 1) > args.diverge_threshold or T - self.last_update_T > 1000:
            self.deploy_net.load_state_dict(self.online_net.state_dict())
            self.num_deploy += 1
            self.last_update_T = T
        self.train()
      elif self.deploy_policy == 'policy' or self.deploy_policy == 'reset_policy':
        self.eval()
        with torch.no_grad():
          deploy_action = (self.deploy_net(states) * self.support).sum(2).argmax(1)
          online_action = (self.online_net(states) * self.support).sum(2).argmax(1)
          diff = 1 - deploy_action.eq(online_action).sum().item() / args.switch_bsz
          self.action_diff.append(diff)
          if diff > args.policy_diff_threshold:
            # if T - self.last_update_T > 1000 or self.deploy_policy == 'policy':
            self.deploy_net.load_state_dict(self.online_net.state_dict())
            self.num_deploy += 1
            self.last_update_T = T
        self.train()


  # Save model parameters on current device (don't move model between devices)
  def save(self, path, name='model.pth'):
    torch.save(self.deploy_net.state_dict(), os.path.join(path, name))

  # Evaluates Q-value based on single state (no batch)
  def evaluate_q(self, state):
    with torch.no_grad():
      return (self.deploy_net(state.unsqueeze(0)) * self.support).sum(2).max(1)[0].item()

  def train(self):
    self.online_net.train()
    self.deploy_net.train()

  def eval(self):
    self.online_net.eval()
    self.deploy_net.eval()
