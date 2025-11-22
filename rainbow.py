import numpy as np, random, math
import torch, torch.nn as nn, torch.nn.functional as F
import torch.optim as optim
from collections import deque, namedtuple


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super().__init__()
        self.in_f, self.out_f = in_features, out_features
        self.weight_mu  = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_eps", torch.zeros(out_features, in_features))

        self.bias_mu  = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_eps", torch.zeros(out_features))

        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        bound = 1 / math.sqrt(self.in_f)
        self.weight_mu.data.uniform_(-bound, bound)
        self.bias_mu.data.uniform_(-bound, bound)

        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_f))
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_f))

    def _f(self, x):
        return x.sign() * torch.sqrt(x.abs())

    def reset_noise(self):
        eps_in  = self._f(torch.randn(self.in_f , device=self.weight_mu.device))
        eps_out = self._f(torch.randn(self.out_f, device=self.weight_mu.device))
        self.weight_eps.copy_(eps_out.ger(eps_in))
        self.bias_eps.copy_(eps_out)

    def forward(self, x):
        if self.training:
            w = self.weight_mu + self.weight_sigma * self.weight_eps
            b = self.bias_mu   + self.bias_sigma   * self.bias_eps
        else:
            w, b = self.weight_mu, self.bias_mu
        return F.linear(x, w, b)


class SumTree:
    def __init__(self, capacity):
        self.N = 1
        while self.N < capacity: self.N <<= 1
        self.tree = np.zeros(2*self.N)
    def update(self, idx, p):
        i = idx + self.N
        self.tree[i] = p
        while i>1:
            i//=2
            self.tree[i] = self.tree[2*i]+self.tree[2*i+1]
    def sample(self, r):
        idx = 1
        while idx < self.N:
            if r <= self.tree[2*idx]:
                idx = 2*idx
            else:
                r -= self.tree[2*idx]
                idx = 2*idx+1
        return idx-self.N
    @property
    def total(self): return self.tree[1]

class PERBuffer:
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=20_000):
        self.cap, self.alpha = capacity, alpha
        self.beta_start, self.beta_frames = beta_start, beta_frames
        self.pos, self.size = 0, 0
        self.data = [None]*capacity
        self.priorities = SumTree(capacity)
        self.max_p = 1.0

    def push(self, *args):
        self.data[self.pos] = args
        self.priorities.update(self.pos, self.max_p ** self.alpha)
        self.pos = (self.pos+1) % self.cap
        self.size = min(self.size+1, self.cap)

    def sample(self, batch_size, frame_idx):
        beta = self.beta_start + frame_idx * (1.0 - self.beta_start)/self.beta_frames
        beta = min(1.0, beta)

        indices, weights, batch = [], [], []
        for _ in range(batch_size):
            r = np.random.rand() * self.priorities.total
            idx = self.priorities.sample(r)
            if idx >= self.size or self.data[idx] is None:
                continue
            prob = self.priorities.tree[idx+self.priorities.N] / self.priorities.total
            w = (self.size * prob) ** (-beta)
            indices.append(idx); weights.append(w); batch.append(self.data[idx])
        weights = torch.tensor(weights, dtype=torch.float32)
        return indices, weights, batch

    def update_priorities(self, indices, td_errors):
        for idx, err in zip(indices, td_errors):
            p = (abs(err)+1e-6) ** self.alpha
            self.priorities.update(idx, p)
            self.max_p = max(self.max_p, p)


class RainbowNet(nn.Module):
    def __init__(self, state_dim, action_dim, atom_n=51, v_min=-10, v_max=10):
        super().__init__()
        self.action_dim = action_dim
        self.atom_n, self.v_min, self.v_max = atom_n, v_min, v_max
        self.delta_z = (v_max-v_min)/(atom_n-1)
        self.fc1 = NoisyLinear(state_dim, 128)
        self.fc2 = NoisyLinear(128, 128)
        self.V  = NoisyLinear(128, atom_n)
        self.A  = NoisyLinear(128, action_dim*atom_n)

    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        val = self.V(x).view(-1,1,self.atom_n)
        adv = self.A(x).view(-1,self.action_dim,self.atom_n)
        q_atoms = val + adv - adv.mean(1, keepdim=True)
        prob = F.softmax(q_atoms, dim=2)
        return prob


Transition = namedtuple("Transition",
                        ("state","action","reward","next_state","done"))

class RainbowAgent:
    def __init__(self, state_dim, action_dim,
                 atom_n=51, v_min=-10, v_max=10,
                 lr=1e-4, gamma=0.99,
                 n_step=3, buffer_size=100_000,
                 batch_size=64, update_target=1000,
                 device="cuda" if torch.cuda.is_available() else "cpu"):

        self.device = torch.device(device)
        self.atom_n, self.v_min, self.v_max = atom_n, v_min, v_max
        self.delta_z = (v_max-v_min)/(atom_n-1)
        self.batch_size, self.gamma = batch_size, gamma
        self.n_step, self.update_target = n_step, update_target
        self.frame_idx = 0

        self.policy_net = RainbowNet(state_dim, action_dim,
                                     atom_n, v_min, v_max).to(self.device)
        self.target_net = RainbowNet(state_dim, action_dim,
                                     atom_n, v_min, v_max).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        self.memory = PERBuffer(buffer_size)
        self.nstep_buffer = deque(maxlen=n_step)

        self.register_buffer = lambda name, t: setattr(self, name, t)
        self.z_atoms = torch.linspace(v_min, v_max, atom_n).to(self.device)

    def _get_n_step(self):
        R, next_s, done = 0, None, 0
        for idx, (s,a,r,s2,d) in enumerate(self.nstep_buffer):
            R += (self.gamma**idx)*r
            next_s, done = s2, d
            if d: break
        return self.nstep_buffer[0][0], self.nstep_buffer[0][1], R, next_s, done
    
    def select_action(self, state, valid_actions=None, training=True):
        """
        valid_actions : list[int]，若为 None 则默认全部动作都有效
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            prob = self.policy_net(state_t)
            q = torch.sum(prob * self.z_atoms, dim=2)[0]
            q = q.cpu().numpy()

        if valid_actions is not None:
            mask = np.ones_like(q, dtype=bool)
            mask[valid_actions] = False
            q[mask] = -1e9

        action = int(np.argmax(q))
        return action

    def append(self, *args):
        self.nstep_buffer.append(args)
        if len(self.nstep_buffer) < self.n_step: return
        transition = self._get_n_step()
        self.memory.push(*transition)

    def train_step(self):
        if self.memory.size < self.batch_size: return None
        self.frame_idx += 1
        idxs, weights, batch = self.memory.sample(self.batch_size, self.frame_idx)
        batch = Transition(*zip(*batch))

        s  = torch.FloatTensor(batch.state).to(self.device)
        a  = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
        r  = torch.FloatTensor(batch.reward).to(self.device)
        ns = torch.FloatTensor(batch.next_state).to(self.device)
        d  = torch.FloatTensor(batch.done).to(self.device)
        weights = weights.to(self.device)

        with torch.no_grad():
            prob_next = self.policy_net(ns)
            q_next = torch.sum(prob_next*self.z_atoms, dim=2)
            a_next = torch.argmax(q_next, dim=1, keepdim=True)

            prob_target_next = self.target_net(ns)
            prob_target_next = prob_target_next[range(self.batch_size),
                                               a_next.squeeze()]
            Tz = (r.unsqueeze(1) +
                  (1-d.unsqueeze(1))* (self.gamma**self.n_step) * self.z_atoms)
            Tz = Tz.clamp(self.v_min, self.v_max)
            b  = (Tz - self.v_min)/self.delta_z
            l  = b.floor().long()
            u  = b.ceil().long()

            proj_dist = torch.zeros_like(prob_target_next)
            for i in range(self.batch_size):
                for j in range(self.atom_n):
                    lj, uj = l[i,j], u[i,j]
                    if lj==uj:
                        proj_dist[i,lj] += prob_target_next[i,j]
                    else:
                        proj_dist[i,lj] += prob_target_next[i,j]*(uj-b[i,j])
                        proj_dist[i,uj] += prob_target_next[i,j]*(b[i,j]-lj)

        prob = self.policy_net(s)
        log_p = torch.log( prob[range(self.batch_size), a.squeeze()] + 1e-6 )
        loss = -(proj_dist * log_p).sum(dim=1) * weights
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

        self.memory.update_priorities(idxs, loss.detach().cpu().numpy())

        if self.frame_idx % self.update_target == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.policy_net.reset_noise()
        self.target_net.reset_noise()
        return loss.mean().item()

    def save(self, path): torch.save(self.policy_net.state_dict(), path)
    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.policy_net.state_dict())
