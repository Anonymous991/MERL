import numpy as np
import random
import torch
from torch.multiprocessing import Manager
from core.mod_utils import compute_stats


class Buffer():
	"""Cyclic Buffer stores experience tuples from the rollouts
		Parameters:
			capacity (int): Maximum number of experiences to hold in cyclic buffer
		"""

	def __init__(self, capacity, buffer_gpu, filter_c=None):
		self.capacity = capacity; self.buffer_gpu = buffer_gpu; self.filter_c = filter_c
		self.manager = Manager()
		self.tuples = self.manager.list() #Temporary shared buffer to get experiences from processes
		self.s = []; self.ns = []; self.a = []; self.r = []; self.done = []

		self.pg_frames = 0; self.total_frames = 0

		#Priority indices
		self.top_r = None
		self.top_g = None

		#Stats
		self.rstats = {'min': None, 'max': None, 'mean': None, 'std': None}
		self.gstats = {'min': None, 'max': None, 'mean': None, 'std': None}



	def data_filter(self, exp):

		# if save_data:
		self.s.append(exp[0])
		self.ns.append(exp[1])
		self.a.append(exp[2])
		self.r.append(exp[3])
		self.done.append(exp[4])
		self.pg_frames += 1
		self.total_frames += 1


	def referesh(self):
		"""Housekeeping
			Parameters:
				None
			Returns:
				None
		"""

		# Add ALL EXPERIENCE COLLECTED TO MEMORY concurrently
		for _ in range(len(self.tuples)):
			exp = self.tuples.pop()
			self.data_filter(exp)


		#Trim to make the buffer size < capacity
		while self.__len__() > self.capacity:
			self.s.pop(0); self.ns.pop(0); self.a.pop(0); self.r.pop(0); self.done.pop(0)


	def __len__(self):
		return len(self.s)

	def sample(self, batch_size, pr_rew=0.0, pr_global=0.0 ):
		"""Sample a batch of experiences from memory with uniform probability
			   Parameters:
				   batch_size (int): Size of the batch to sample
			   Returns:
				   Experience (tuple): A tuple of (state, next_state, action, shaped_reward, done) each as a numpy array with shape (batch_size, :)
		   """
		#Uniform sampling
		ind = random.sample(range(len(self.s)), batch_size)

		if pr_global != 0.0 or pr_rew !=0.0:
			#Prioritization
			num_r = int(pr_rew * batch_size); num_global = int(pr_global * batch_size)
			ind_r = random.sample(self.top_r, num_r)
			ind_global = random.sample(self.top_g, num_global)

			ind = ind[num_r+num_global:] + ind_r + ind_global


		#return self.sT[ind], self.nsT[ind], self.aT[ind], self.rT[ind], self.doneT[ind], self.global_rewardT[ind]
		return torch.Tensor(np.vstack([self.s[i] for i in ind])), \
			   torch.Tensor(np.vstack([self.ns[i] for i in ind])),\
			   torch.Tensor(np.vstack([self.a[i] for i in ind])),\
			   torch.Tensor(np.vstack([self.r[i] for i in ind])), \
			   torch.Tensor(np.vstack([self.done[i] for i in ind]))


	def tensorify(self):
		"""Method to save experiences to drive
			   Parameters:
				   None
			   Returns:
				   None
		   """
		self.referesh() #Referesh first


