
from core.mod_utils import pprint, str2bool
import numpy as np, os, time, torch
import core.mod_utils as mod
import argparse
from core.models import MultiHeadActor
from core import mod_utils as utils
import numpy as np, random, sys
from envs.env_wrapper import RoverDomainPython
import copy
from functools import partial

from pytorch_es import EvolutionModule


parser = argparse.ArgumentParser()
parser.add_argument('-popsize', type=int, help='#Evo Population size', default=100)
parser.add_argument('-rollsize', type=int, help='#Rollout size for agents', default=0)
parser.add_argument('-env', type=str, help='Env to test on?', default='rover_tight')
parser.add_argument('-config', type=str, help='World Setting?', default='6_3')
parser.add_argument('-reward', type=str, help='Reward Structure? 1. mixed 2. global', default='global')
parser.add_argument('-frames', type=float, help='Frames in millions?', default=2)
parser.add_argument('-seed', type=int, help='#Seed', default=2019)
parser.add_argument('-savetag', help='Saved tag', default='ES_')

parser.add_argument('-sigma', type=float, help='Saved tag', default=0.1)
parser.add_argument('-lr', type=float, help='Saved tag', default=1e-3)
RANDOM_BASELINE = False





#Rollout evaluate an agent in a complete game
def evaluate(weights, args, NUM_EVALS=10, render=False):
	"""Rollout Worker runs a simulation in the environment to generate experiences and fitness values

		Parameters:
			args (object): Parameter class
			id (int): Specific Id unique to each worker spun
			task_pipe (pipe): Receiver end of the task pipe used to receive signal to start on a task
			result_pipe (pipe): Sender end of the pipe used to report back results
			data_bucket (shared list object): A shared list object managed by a manager that is used to store experience tuples
			models_bucket (shared list object): A shared list object managed by a manager used to store all the models (actors)
			store_transition (bool): Log experiences to exp_list?
			random_baseline (bool): Test with random action for baseline purpose?

		Returns:
			None
	"""
	env = RoverDomainPython(args, NUM_EVALS)
	model = MultiHeadActor(args.state_dim, args.action_dim, args.hidden_size, args.config.num_agents)

	for i, param in enumerate(model.parameters()):
		try:
			param.data = weights[i]
		except:
			param.data = weights[i].data



	fitness = [None for _ in range(NUM_EVALS)]; frame=0
	joint_state = env.reset()
	joint_state = utils.to_tensor(np.array(joint_state))

	while True: #unless done

		joint_action = [model.clean_action(joint_state[i, :], head=i).detach().numpy() for i in range(args.config.num_agents)]

		#JOINT ACTION [agent_id, universe_id, action]
		#Bound Action
		joint_action = np.array(joint_action).clip(-1.0, 1.0)
		next_state, reward, done, global_reward = env.step(joint_action)  # Simulate one step in environment
		#State --> [agent_id, universe_id, obs]
		#reward --> [agent_id, universe_id]
		#done --> [universe_id]
		#info --> [universe_id]

		next_state = utils.to_tensor(np.array(next_state))

		#Grab global reward as fitnesses
		for i, grew in enumerate(global_reward):
			if grew != None:
				fitness[i] = grew


		joint_state = next_state
		frame+=NUM_EVALS

		#DONE FLAG IS Received
		if sum(done)==len(done):
			break

	return sum(fitness)/len(fitness)


class ConfigSettings:
	def __init__(self):

		self.env_choice = vars(parser.parse_args())['env']
		config = vars(parser.parse_args())['config']
		self.config = config
		self.reward_scheme = vars(parser.parse_args())['reward']

		self.is_lsg = False
		self.is_proxim_rew = True




		# ROVER DOMAIN
		if self.env_choice == 'rover_tight':  # Rover Domain


			##########LOOSE##########
			if config == '3_1':
				# Rover domain
				self.dim_x = self.dim_y = 30; self.obs_radius = self.dim_x * 10; self.act_dist = 3; self.rover_speed = 1; self.sensor_model = 'closest'
				self.angle_res = 10
				self.num_poi = 3
				self.num_agents = 3
				self.ep_len = 50
				self.poi_rand = 1
				self.coupling = 1

			##########TIGHT##########
			elif config == '4_2':
				# Rover domain
				self.dim_x = self.dim_y = 20;
				self.obs_radius = self.dim_x * 10;
				self.act_dist = 3;
				self.rover_speed = 1;
				self.sensor_model = 'closest'
				self.angle_res = 10
				self.num_poi = 4
				self.num_agents = 4
				self.ep_len = 50
				self.poi_rand = 1
				self.coupling = 2

			elif config == '6_3':
				# Rover domain
				self.dim_x = self.dim_y = 20; self.obs_radius = self.dim_x * 10; self.act_dist = 3; self.rover_speed = 1; self.sensor_model = 'closest'
				self.angle_res = 10
				self.num_poi = 4
				self.num_agents = 6
				self.ep_len = 50
				self.poi_rand = 1
				self.coupling = 3

			elif config == '8_4':
				# Rover domain
				self.dim_x = self.dim_y = 20; self.obs_radius = self.dim_x * 10; self.act_dist = 3; self.rover_speed = 1; self.sensor_model = 'closest'
				self.angle_res = 10
				self.num_poi = 4
				self.num_agents = 8
				self.ep_len = 50
				self.poi_rand = 1
				self.coupling = 4

			elif config == '10_5':
				# Rover domain
				self.dim_x = self.dim_y = 20; self.obs_radius = self.dim_x * 10; self.act_dist = 3; self.rover_speed = 1; self.sensor_model = 'closest'
				self.angle_res = 10
				self.num_poi = 4
				self.num_agents = 10
				self.ep_len = 50
				self.poi_rand = 1
				self.coupling = 5

			elif config == '12_6':
				# Rover domain
				self.dim_x = self.dim_y = 20; self.obs_radius = self.dim_x * 10; self.act_dist = 3; self.rover_speed = 1; self.sensor_model = 'closest'
				self.angle_res = 10
				self.num_poi = 4
				self.num_agents = 12
				self.ep_len = 50
				self.poi_rand = 1
				self.coupling = 6

			elif config == '14_7':
				# Rover domain
				self.dim_x = self.dim_y = 20; self.obs_radius = self.dim_x * 10; self.act_dist = 3; self.rover_speed = 1; self.sensor_model = 'closest'
				self.angle_res = 10
				self.num_poi = 4
				self.num_agents = 14
				self.ep_len = 50
				self.poi_rand = 1
				self.coupling = 7



			else:
				sys.exit('Unknown Config')

			#Fix Harvest Period and coupling given some config choices

			if self.env_choice == "rover_trap": self.harvest_period = 3
			else: self.harvest_period = 1

			if self.env_choice == "rover_loose": self.coupling = 1 #Definiton of a Loosely coupled domain


		elif self.env_choice == 'motivate':  # Rover Domain
			# Motivate domain
			self.dim_x = self.dim_y = 20
			self.obs_radius = self.dim_x * 10
			self.act_dist = 2
			self.angle_res = 10
			self.num_poi = 2
			self.num_agents = 2
			self.ep_len = 30
			self.poi_rand = 0
			self.coupling = 1
			self.rover_speed = 1
			self.sensor_model = 'closest'
			self.harvest_period = 1

		else:
			sys.exit('Unknown Environment Choice')


class Parameters:
	def __init__(self):

		# Transitive Algo Params
		self.popsize = vars(parser.parse_args())['popsize']
		self.seed = vars(parser.parse_args())['seed']
		self.sigma = vars(parser.parse_args())['sigma']
		self.lr = vars(parser.parse_args())['lr']


		# Env domain
		self.config = ConfigSettings()
		self.hidden_size = 100

		# Fairly Stable Algo params


		# Dependents
		self.state_dim = int(720 / self.config.angle_res) + 3
		self.action_dim = 2


		# Save Filenames
		self.savetag = vars(parser.parse_args())['savetag'] + \
		               '_popsize' + str(self.popsize) + \
		               '_env' + str(self.config.env_choice) + '_' + str(self.config.config) + \
					   '_sigma' + str(self.sigma) + \
					   '_lr' + str(self.lr) + \
					   '_seed' + str(self.seed)

		self.save_foldername = 'R_MERL/'
		if not os.path.exists(self.save_foldername): os.makedirs(self.save_foldername)
		self.metric_save = self.save_foldername + 'metrics/'
		self.model_save = self.save_foldername + 'models/'
		self.aux_save = self.save_foldername + 'auxiliary/'
		if not os.path.exists(self.save_foldername): os.makedirs(self.save_foldername)
		if not os.path.exists(self.metric_save): os.makedirs(self.metric_save)
		if not os.path.exists(self.model_save): os.makedirs(self.model_save)
		if not os.path.exists(self.aux_save): os.makedirs(self.aux_save)

		self.critic_fname = 'critic_' + self.savetag
		self.actor_fname = 'actor_' + self.savetag
		self.log_fname = 'reward_' + self.savetag
		self.best_fname = 'best_' + self.savetag





if __name__ == "__main__":
	args = Parameters()  # Create the Parameters class


	test_tracker = utils.Tracker(args.metric_save, [args.log_fname], '.csv')  # Initiate tracker
	torch.manual_seed(args.seed);
	np.random.seed(args.seed);
	random.seed(args.seed)  # Seeds

	model = MultiHeadActor(args.state_dim, args.action_dim, args.hidden_size, args.config.num_agents)
	# EvolutionModule runs the population in a ThreadPool, so
	# if you need to inject other arguments, you can do that
	# using the partial tool
	partial_func = partial(evaluate, args=args)
	mother_parameters = list(model.parameters())

	es = EvolutionModule(
		mother_parameters, partial_func, population_size=args.popsize,
		sigma=args.sigma, learning_rate=args.lr,
		threadcount=100, cuda=False, render_test=False
	)

	###### TRAINING LOOP ########
	total_frames = 0
	while True:
		#Train
		es.run(iterations=10, print_step = 100)

		total_frames += 10*args.popsize*args.config.ep_len*10

		test_score = evaluate(es.weights, args, NUM_EVALS=50, render=False)
		test_tracker.update([test_score], total_frames)

		print('Total_Frames', total_frames, 'Test_Score', test_score)






