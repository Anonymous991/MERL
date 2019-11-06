import numpy as np, sys



class SimpleTag:
	"""Wrapper around the Environment to expose a cleaner interface for RL

		Parameters:
			env_name (str): Env name


	"""
	def __init__(self, args, num_envs=1):
		"""
		A base template for all environment wrappers.
		"""
		#Initialize world with requiste params
		self.args = args
		self.num_envs = num_envs
		self.i = 0
		self.T = 25

		from envs.maddpg_envs.make_env import make_env

		self.universe = [] #Universe - collection of all envs running in parallel
		for _ in range(num_envs):
			env = make_env(args.config.config)
			self.universe.append(env)

		self.global_reward = [[0.0,0.0] for _ in range(num_envs)]



	def reset(self):
		"""Method overloads reset
			Parameters:
				None

			Returns:
				next_obs (list): Next state
		"""
		#Reset Global Reward and dones
		self.global_reward = [[0.0,0.0] for _ in range(self.num_envs)]
		self.i = 0

		#Get joint observation
		predator_obs = []; prey_obs = []
		for env in self.universe:
			obs = env.reset()
			prey_obs.append(obs[3:6])
			predator_obs.append(obs[0:3])

		prey_obs = np.stack(prey_obs, axis=1)
		predator_obs = np.stack(predator_obs, axis=1)
		return predator_obs, prey_obs


	def step(self, pred_action, prey_action): #Expects a numpy action
		"""Take an action to forward the simulation

			Parameters:
				action (ndarray): action to take in the env

			Returns:
				next_obs (list): Next state
				reward (float): Reward for this step
				done (bool): Simulation done?
				info (None): Template from OpenAi gym (doesnt have anything)
		"""
		joint_action = np.concatenate((pred_action, prey_action), axis=0)

		pred_obs = []; pred_reward = []
		prey_obs = []; prey_reward = []
		self.i+=1

		for universe_id, env in enumerate(self.universe):

			#If this particular env instance in universe is already done:
			next_state, reward, _, _ = env.step(joint_action[:,universe_id,:])
			done = self.i > self.T
			prey_obs.append(next_state[3:6])
			pred_obs.append(next_state[0:3])
			pred_reward.append(reward[0:3])
			prey_reward.append(reward[3:6])
			self.global_reward[universe_id][0] = env.world.num_collisions
			self.global_reward[universe_id][1] += sum(prey_reward[-1]) / (self.T)


		pred_obs = np.stack(pred_obs, axis=1)
		prey_obs = np.stack(prey_obs, axis=1)
		pred_reward = np.stack(pred_reward, axis=1)
		prey_reward = np.stack(prey_reward, axis=1)


		return pred_obs, prey_obs, pred_reward, prey_reward, done, self.global_reward if done else [[None,None] for _ in range(self.num_envs)]



	def render(self, env_id=None):
		if env_id == None:
			rand_univ = np.random.randint(0, len(self.universe))
		else: rand_univ = env_id

		self.universe[rand_univ].render()

class SimpleAdversary:
	"""Wrapper around the Environment to expose a cleaner interface for RL

		Parameters:
			env_name (str): Env name


	"""
	def __init__(self, args, num_envs=1):
		"""
		A base template for all environment wrappers.
		"""
		#Initialize world with requiste params
		self.args = args
		self.num_envs = num_envs
		self.i = 0
		self.T = 25

		from envs.maddpg_envs.make_env import make_env

		self.universe = [] #Universe - collection of all envs running in parallel
		for _ in range(num_envs):
			env = make_env(args.config.config)
			self.universe.append(env)

		self.global_reward = [[0.0, 0.0] for _ in range(num_envs)]



	def reset(self):
		"""Method overloads reset
			Parameters:
				None

			Returns:
				next_obs (list): Next state
		"""
		#Reset Global Reward and dones
		self.global_reward = [[0.0,0.0] for _ in range(self.num_envs)]
		self.i = 0

		#Get joint observation
		predator_obs = []; prey_obs = []
		for env in self.universe:
			obs = env.reset()
			predator_obs.append(obs[1:3])
			prey_obs.append(obs[0:1])

		predator_obs = np.stack(predator_obs, axis=1)
		prey_obs = np.stack(prey_obs, axis=1)
		return predator_obs, prey_obs


	def step(self, pred_action, prey_action): #Expects a numpy action
		"""Take an action to forward the simulation

			Parameters:
				action (ndarray): action to take in the env

			Returns:
				next_obs (list): Next state
				reward (float): Reward for this step
				done (bool): Simulation done?
				info (None): Template from OpenAi gym (doesnt have anything)
		"""
		joint_action = np.concatenate((pred_action, prey_action), axis=0)

		pred_obs = []; pred_reward = []
		prey_obs = []; prey_reward = []
		self.i+=1

		for universe_id, env in enumerate(self.universe):

			#If this particular env instance in universe is already done:
			next_state, reward, _, _ = env.step(joint_action[:,universe_id,:])
			done = self.i > self.T
			prey_obs.append(next_state[0:1])
			pred_obs.append(next_state[1:3])

			prey_reward.append(reward[0:1])
			pred_reward.append(reward[1:3])

			self.global_reward[universe_id][0] += sum(pred_reward[-1])/(self.T*len(pred_reward[-1]))
			self.global_reward[universe_id][1] += sum(prey_reward[-1])/(self.T*len(prey_reward[-1]))


		pred_obs = np.stack(pred_obs, axis=1)
		prey_obs = np.stack(prey_obs, axis=1)
		pred_reward = np.stack(pred_reward, axis=1)
		prey_reward = np.stack(prey_reward, axis=1)


		return pred_obs, prey_obs, pred_reward, prey_reward, done, self.global_reward if done else [[None,None] for _ in range(self.num_envs)]



	def render(self, env_id=None):
		if env_id == None:
			rand_univ = np.random.randint(0, len(self.universe))
		else: rand_univ = env_id

		self.universe[rand_univ].render()


class SimplePush:
	"""Wrapper around the Environment to expose a cleaner interface for RL

		Parameters:
			env_name (str): Env name


	"""
	def __init__(self, args, num_envs=1):
		"""
		A base template for all environment wrappers.
		"""
		#Initialize world with requiste params
		self.args = args
		self.num_envs = num_envs
		self.i = 0
		self.T = 25

		from envs.maddpg_envs.make_env import make_env

		self.universe = [] #Universe - collection of all envs running in parallel
		for _ in range(num_envs):
			env = make_env(args.config.config)
			self.universe.append(env)

		self.global_reward = [[0.0, 0.0] for _ in range(num_envs)]



	def reset(self):
		"""Method overloads reset
			Parameters:
				None

			Returns:
				next_obs (list): Next state
		"""
		#Reset Global Reward and dones
		self.global_reward = [[0.0,0.0] for _ in range(self.num_envs)]
		self.i = 0

		#Get joint observation
		predator_obs = []; prey_obs = []
		for env in self.universe:
			obs = env.reset()
			predator_obs.append(obs[1:3])
			prey_obs.append(obs[0:1])

		predator_obs = np.stack(predator_obs, axis=1)
		prey_obs = np.stack(prey_obs, axis=1)
		return predator_obs, prey_obs


	def step(self, pred_action, prey_action): #Expects a numpy action
		"""Take an action to forward the simulation

			Parameters:
				action (ndarray): action to take in the env

			Returns:
				next_obs (list): Next state
				reward (float): Reward for this step
				done (bool): Simulation done?
				info (None): Template from OpenAi gym (doesnt have anything)
		"""
		joint_action = np.concatenate((pred_action, prey_action), axis=0)

		pred_obs = []; pred_reward = []
		prey_obs = []; prey_reward = []
		self.i+=1

		for universe_id, env in enumerate(self.universe):

			#If this particular env instance in universe is already done:
			next_state, reward, _, _ = env.step(joint_action[:,universe_id,:])
			done = self.i > self.T
			prey_obs.append(next_state[0:1])
			pred_obs.append(next_state[1:3])

			prey_reward.append(reward[0:1])
			pred_reward.append(reward[1:3])

			self.global_reward[universe_id][0] += sum(pred_reward[-1])/(self.T*len(pred_reward[-1]))
			self.global_reward[universe_id][1] += sum(prey_reward[-1])/(self.T*len(prey_reward[-1]))


		pred_obs = np.stack(pred_obs, axis=1)
		prey_obs = np.stack(prey_obs, axis=1)
		pred_reward = np.stack(pred_reward, axis=1)
		prey_reward = np.stack(prey_reward, axis=1)


		return pred_obs, prey_obs, pred_reward, prey_reward, done, self.global_reward if done else [[None,None] for _ in range(self.num_envs)]



	def render(self, env_id=None):
		if env_id == None:
			rand_univ = np.random.randint(0, len(self.universe))
		else: rand_univ = env_id

		self.universe[rand_univ].render()