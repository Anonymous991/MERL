import torch, os
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from core import mod_utils as utils
from core.models import QNetwork, MultiHeadActor


class MultiTD3(object):
	"""Classes implementing TD3 and DDPG off-policy learners




	 """
	def __init__(self, id, algo_name, state_dim, action_dim, hidden_size, actor_lr, critic_lr, gamma, tau, savetag, foldername, use_gpu, num_agents, init_w = True):

		self.algo_name = algo_name; self.gamma = gamma; self.tau = tau; self.total_update = 0; self.agent_id = id; self.use_gpu = use_gpu
		self.tracker = utils.Tracker(foldername, ['q_'+savetag, 'qloss_'+savetag, 'policy_loss_'+savetag], '.csv', save_iteration=1000, conv_size=1000)

		#Initialize actors
		self.policy = MultiHeadActor(state_dim, action_dim, hidden_size, num_agents)
		if init_w: self.policy.apply(utils.init_weights)
		self.policy_target = MultiHeadActor(state_dim, action_dim, hidden_size, num_agents)
		if use_gpu:
			self.policy.cuda()
			self.policy_target.cuda()
		utils.hard_update(self.policy_target, self.policy)
		self.policy_optim = Adam(self.policy.parameters(), actor_lr)


		self.critics = [QNetwork(state_dim, action_dim, hidden_size) for _ in range(num_agents)]

		self.critics_target = [QNetwork(state_dim, action_dim, hidden_size) for _ in range(num_agents)]
		if init_w:
			for critic, critic_target in zip(self.critics, self.critics_target):
				critic.apply(utils.init_weights)
				utils.hard_update(critic_target, critic)
				if use_gpu:
					critic.cuda();
					critic_target.cuda();
		self.critic_optims = [Adam(critic.parameters(), critic_lr) for critic in self.critics]


		self.loss = nn.MSELoss()


		self.num_critic_updates = 0

		#Statistics Tracker
		self.policy_loss = {'min':None, 'max': None, 'mean':None, 'std':None}
		self.q_loss = {'min':None, 'max': None, 'mean':None, 'std':None}
		self.q = {'min':None, 'max': None, 'mean':None, 'std':None}



	def update_parameters(self, state_batch, next_state_batch, action_batch, reward_batch, done_batch, global_reward, agent_id, num_epoch=1, **kwargs):
		"""Runs a step of Bellman upodate and policy gradient using a batch of experiences



		 """

		if isinstance(state_batch, list): state_batch = torch.cat(state_batch); next_state_batch = torch.cat(next_state_batch); action_batch = torch.cat(action_batch); reward_batch = torch.cat(reward_batch). done_batch = torch.cat(done_batch); global_reward = torch.cat(global_reward)

		for _ in range(num_epoch):
			########### CRITIC UPDATE ####################

			#Compute next q-val, next_v and target
			with torch.no_grad():
				
				#Policy Noise
				policy_noise = np.random.normal(0, kwargs['policy_noise'], (action_batch.size()[0], action_batch.size()[1]))
				policy_noise = torch.clamp(torch.Tensor(policy_noise), -kwargs['policy_noise_clip'], kwargs['policy_noise_clip'])

				#Compute next action_bacth
				next_action_batch = self.policy_target.clean_action(next_state_batch, agent_id) + policy_noise.cuda() if self.use_gpu else policy_noise
				next_action_batch = torch.clamp(next_action_batch, -1, 1)

				#Compute Q-val and value of next state masking by done
				q1, q2 = self.critics_target[agent_id].forward(next_state_batch, next_action_batch)
				q1 = (1 - done_batch) * q1
				q2 = (1 - done_batch) * q2

				#Select which q to use as next-q (depends on algo)
				if self.algo_name == 'TD3': next_q = torch.min(q1, q2)
				elif self.algo_name == 'DDPG': next_q = q1

				#Compute target q and target val
				target_q = reward_batch + (self.gamma * next_q)



			self.critic_optims[agent_id].zero_grad()
			current_q1, current_q2 = self.critics[agent_id].forward((state_batch), (action_batch))
			utils.compute_stats(current_q1, self.q)

			dt = self.loss(current_q1, target_q)

			if self.algo_name == 'TD3': dt = dt + self.loss(current_q2, target_q)
			utils.compute_stats(dt, self.q_loss)
			dt.backward()

			self.critic_optims[agent_id].step()
			self.num_critic_updates += 1


			#Delayed Actor Update
			if self.num_critic_updates % kwargs['policy_ups_freq'] == 0:

				actor_actions = self.policy.clean_action(state_batch, agent_id)
				Q1, Q2 = self.critics[agent_id].forward(state_batch, actor_actions)

				# if self.args.use_advantage: policy_loss = -(Q1 - val)
				policy_loss = -Q1

				utils.compute_stats(-policy_loss,self.policy_loss)
				policy_loss = policy_loss.mean()

				self.policy_optim.zero_grad()



				policy_loss.backward(retain_graph=True)
				self.policy_optim.step()


			if self.num_critic_updates % kwargs['policy_ups_freq'] == 0: utils.soft_update(self.policy_target, self.policy, self.tau)
			utils.soft_update(self.critics_target[agent_id], self.critics[agent_id], self.tau)

			self.total_update += 1
			if self.agent_id == 0:
				self.tracker.update([self.q['mean'], self.q_loss['mean'], self.policy_loss['mean']] ,self.total_update)



class MATD3(object):
	"""Classes implementing TD3 and DDPG off-policy learners



	 """
	def __init__(self, id, algo_name, state_dim, action_dim, hidden_size, actor_lr, critic_lr, gamma, tau, savetag, foldername, use_gpu, num_agents, init_w = True):

		self.algo_name = algo_name; self.gamma = gamma; self.tau = tau; self.total_update = 0; self.agent_id = id;self.use_gpu = use_gpu
		self.tracker = utils.Tracker(foldername, ['q_'+savetag, 'qloss_'+savetag, 'policy_loss_'+savetag], '.csv', save_iteration=1000, conv_size=1000)
		self.num_agents = num_agents

		#Initialize actors
		self.policy = MultiHeadActor(state_dim, action_dim, hidden_size, num_agents)
		if init_w: self.policy.apply(utils.init_weights)
		self.policy_target = MultiHeadActor(state_dim, action_dim, hidden_size, num_agents)
		utils.hard_update(self.policy_target, self.policy)
		self.policy_optim = Adam(self.policy.parameters(), actor_lr)


		self.critics = [QNetwork(state_dim*num_agents, action_dim*num_agents, hidden_size*3) for _ in range(num_agents)]

		self.critics_target = [QNetwork(state_dim*num_agents, action_dim*num_agents, hidden_size*3) for _ in range(num_agents)]
		if init_w:
			for critic, critic_target in zip(self.critics, self.critics_target):
				critic.apply(utils.init_weights)
				utils.hard_update(critic_target, critic)
		self.critic_optims = [Adam(critic.parameters(), critic_lr) for critic in self.critics]


		self.loss = nn.MSELoss()

		if use_gpu:
			self.policy_target.cuda(); self.policy.cuda()
			for critic, critic_target in zip(self.critics, self.critics_target):
				critic.cuda()
				critic_target.cuda()


		self.num_critic_updates = 0

		#Statistics Tracker
		self.policy_loss = {'min':None, 'max': None, 'mean':None, 'std':None}
		self.q_loss = {'min':None, 'max': None, 'mean':None, 'std':None}
		self.q = {'min':None, 'max': None, 'mean':None, 'std':None}



	def update_parameters(self, state_batch, next_state_batch, action_batch, reward_batch, done_batch, global_reward, agent_id, num_epoch=1, **kwargs):
		"""Runs a step of Bellman upodate and policy gradient using a batch of experiences

			 Parameters:
				  state_batch (tensor): Current States
				  next_state_batch (tensor): Next States
				  action_batch (tensor): Actions
				  reward_batch (tensor): Rewards
				  done_batch (tensor): Done batch
				  num_epoch (int): Number of learning iteration to run with the same data

			 Returns:
				   None

		 """

		if isinstance(state_batch, list): state_batch = torch.cat(state_batch); next_state_batch = torch.cat(next_state_batch); action_batch = torch.cat(action_batch); reward_batch = torch.cat(reward_batch). done_batch = torch.cat(done_batch); global_reward = torch.cat(global_reward)
		batch_size = len(state_batch)

		for _ in range(num_epoch):
			########### CRITIC UPDATE ####################

			#Compute next q-val, next_v and target
			with torch.no_grad():


				#Compute next action_bacth
				next_action_batch = torch.cat([self.policy_target.clean_action(next_state_batch[:, id, :], id) for id in range(self.num_agents)], 1)
				if self.algo_name == 'TD3':
					# Policy Noise
					policy_noise = np.random.normal(0, kwargs['policy_noise'], (action_batch.size()[0], action_batch.size()[1] * action_batch.size()[2]))
					policy_noise = torch.clamp(torch.Tensor(policy_noise), -kwargs['policy_noise_clip'], kwargs['policy_noise_clip'])
					next_action_batch += policy_noise.cuda() if self.use_gpu else policy_noise
				next_action_batch = torch.clamp(next_action_batch, -1, 1)

				#Compute Q-val and value of next state masking by done
				q1, q2 = self.critics_target[agent_id].forward(next_state_batch.view(batch_size, -1), next_action_batch)
				q1 = (1 - done_batch) * q1
				q2 = (1 - done_batch) * q2
				#next_val = (1 - done_batch) * next_val

				#Select which q to use as next-q (depends on algo)
				if self.algo_name == 'TD3':next_q = torch.min(q1, q2)
				elif self.algo_name == 'DDPG': next_q = q1

				#Compute target q and target val
				target_q = reward_batch[:,agent_id].unsqueeze(1) + (self.gamma * next_q)


			self.critic_optims[agent_id].zero_grad()
			current_q1, current_q2 = self.critics[agent_id].forward((state_batch.view(batch_size, -1)), (action_batch.view(batch_size, -1)))
			utils.compute_stats(current_q1, self.q)

			dt = self.loss(current_q1, target_q)

			if self.algo_name == 'TD3': dt = dt + self.loss(current_q2, target_q)
			utils.compute_stats(dt, self.q_loss)
			dt.backward()

			self.critic_optims[agent_id].step()
			self.num_critic_updates += 1

			#Delayed Actor Update
			if self.num_critic_updates % kwargs['policy_ups_freq'] == 0 or self.algo_name == 'DDPG':

				agent_action = self.policy.clean_action(state_batch[:,agent_id,:], agent_id)
				joint_action = action_batch.clone()
				joint_action[:,agent_id,:] = agent_action[:]

				Q1, Q2 = self.critics[agent_id].forward(state_batch.view(batch_size, -1), joint_action.view(batch_size, -1))
				policy_loss = -Q1

				utils.compute_stats(-policy_loss,self.policy_loss)
				policy_loss = policy_loss.mean()

				self.policy_optim.zero_grad()



				policy_loss.backward(retain_graph=True)
				self.policy_optim.step()


			if self.num_critic_updates % kwargs['policy_ups_freq'] == 0 or self.algo_name == 'DDPG': utils.soft_update(self.policy_target, self.policy, self.tau)
			for critic, critic_target in zip(self.critics, self.critics_target):
				utils.soft_update(critic_target, critic, self.tau)

			self.total_update += 1
			if self.agent_id == 0:
				self.tracker.update([self.q['mean'], self.q_loss['mean'], self.policy_loss['mean']] ,self.total_update)
