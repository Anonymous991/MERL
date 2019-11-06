from core.agent import Agent, TestAgent, PreyAgent
from core.mod_utils import pprint, str2bool
import numpy as np, os, time, torch
from core import mod_utils as utils
from core.runner import rollout_worker
from torch.multiprocessing import Process, Pipe
import core.mod_utils as mod
import argparse
import random
import sys



parser = argparse.ArgumentParser()
parser.add_argument('--popsize', type=int, help='#Evo Population size', default=10)
parser.add_argument('--rollsize', type=int, help='#Rollout size for agents', default=10)
parser.add_argument('--env', type=str, help='Env to test on?', default='maddpg_envs')
parser.add_argument('--config', type=str, help='World Setting?', default='simple_tag')
parser.add_argument('--matd3', type=str2bool, help='Use_MATD3?', default=False)
parser.add_argument('--maddpg', type=str2bool, help='Use_MADDPG?', default=False)
parser.add_argument('--reward', type=str, help='Reward Structure? 1. mixed 2. global', default='mixed')
parser.add_argument('--frames', type=float, help='Frames in millions?', default=10)


parser.add_argument('--filter_c', type=int, help='Prob multiplier for evo experiences absorbtion into buffer?', default=1)
parser.add_argument('--evals', type=int, help='#Evals to compute a fitness', default=1)
parser.add_argument('--seed', type=int, help='#Seed', default=2018)
parser.add_argument('--algo', type=str, help='SAC Vs. TD3?', default='TD3')
parser.add_argument('--savetag', help='Saved tag', default='new_tag_')
parser.add_argument('--gradperstep', type=float, help='gradient steps per frame', default=1.0)
parser.add_argument('--pr', type=float, help='Prioritization?', default=0.0)
parser.add_argument('--gpu_id', type=int, help='USE_GPU?', default=0)
parser.add_argument('--alz', type=str2bool, help='Actualize?', default=False)
parser.add_argument('--scheme', type=str, help='Scheme?', default='standard')
parser.add_argument('--cmd_vel', type=str2bool, help='Switch to Velocity commands?', default=True)
parser.add_argument('--ps', type=str, help='Parameter Sharing Scheme: 1. none (heterogenous) 2. full (homogeneous) 3. trunk (shared trunk - similar to multi-headed)?', default='trunk')

RANDOM_BASELINE = False
if vars(parser.parse_args())['gpu_id'] != -1: os.environ["CUDA_VISIBLE_DEVICES"]=str(vars(parser.parse_args())['gpu_id'])


class ConfigSettings:
	def __init__(self, popnsize):

		self.env_choice = vars(parser.parse_args())['env']
		config = vars(parser.parse_args())['config']
		self.config = config
		#Global subsumes local or vice-versa?
		####################### NIPS EXPERIMENTS SETUP #################
		self.cmd_vel = vars(parser.parse_args())['cmd_vel']
		if self.env_choice == 'maddpg_envs':
			if self.config == 'simple_tag' or self.config == 'hard_tag':
				self.num_agents = 3
			elif self.config == 'simple_adversary':
				self.num_agents = 2
			elif self.config == 'simple_push':
				self.num_agents = 2
			else:
				sys.exit('Unknown Config Choice')


		else:
			sys.exit('Unknown Environment Choice')

class Parameters:
	def __init__(self):

		# Transitive Algo Params
		self.popn_size = vars(parser.parse_args())['popsize']
		self.rollout_size = vars(parser.parse_args())['rollsize']
		self.num_evals = vars(parser.parse_args())['evals']
		self.frames_bound = int(vars(parser.parse_args())['frames'] * 1000000)
		self.actualize = vars(parser.parse_args())['alz']
		self.priority_rate = vars(parser.parse_args())['pr']
		self.use_gpu = torch.cuda.is_available()
		self.seed = vars(parser.parse_args())['seed']
		self.ps = vars(parser.parse_args())['ps']
		self.is_matd3 = vars(parser.parse_args())['matd3']
		self.is_maddpg = vars(parser.parse_args())['maddpg']
		assert  self.is_maddpg * self.is_matd3 == 0 #Cannot be both True

		# Env domain
		self.config = ConfigSettings(self.popn_size)

		# Fairly Stable Algo params
		self.hidden_size = 100
		self.algo_name = vars(parser.parse_args())['algo']
		self.actor_lr = 5e-5
		self.critic_lr = 1e-5
		self.tau = 1e-5
		self.init_w = True
		self.gradperstep = vars(parser.parse_args())['gradperstep']
		self.gamma = 0.5 if self.popn_size > 0 else 0.97
		self.batch_size = 512
		self.buffer_size = 100000
		self.filter_c = vars(parser.parse_args())['filter_c']
		self.reward_scaling = 10.0

		self.action_loss = False
		self.policy_ups_freq = 2
		self.policy_noise = True
		self.policy_noise_clip = 0.4

		# SAC
		self.alpha = 0.2
		self.target_update_interval = 1

		# NeuroEvolution stuff
		self.scheme = vars(parser.parse_args())['scheme']  # 'multipoint' vs 'standard'
		self.crossover_prob = 0.1
		self.mutation_prob = 0.9
		self.extinction_prob = 0.005  # Probability of extinction event
		self.extinction_magnitude = 0.5  # Probabilty of extinction for each genome, given an extinction event
		self.weight_clamp = 1000000
		self.mut_distribution = 1  # 1-Gaussian, 2-Laplace, 3-Uniform
		self.lineage_depth = 10
		self.ccea_reduction = "leniency"
		self.num_anchors = 5
		self.num_elites = 4
		self.num_blends = int(0.15 * self.popn_size)

		# Dependents
		self.hidden_size = 100
		self.actor_lr = 0.01
		self.critic_lr = 0.01
		self.tau = 0.01
		self.init_w = True
		self.gradperstep = vars(parser.parse_args())['gradperstep']
		self.gamma = 0.95
		self.batch_size = 1024
		self.buffer_size = 1000000
		self.reward_scaling = 1.0
		self.num_test = 10
		self.test_gap = 5
		self.action_dim = 2

		if self.config.config == 'simple_tag' or self.config.config == 'hard_tag':
			self.pred_state_dim = 26
			self.prey_state_dim = 24

		elif self.config.config == 'simple_adversary':
			self.pred_state_dim = 10
			self.prey_state_dim = 8

		elif self.config.config == 'simple_push':
			self.pred_state_dim = 21
			self.prey_state_dim = 10
		else:
			sys.exit('Unknow Config')



		# Save Filenames
		self.savetag = vars(parser.parse_args())['savetag'] + \
		               'pop' + str(self.popn_size) + \
		               '_roll' + str(self.rollout_size) + \
		               '_env' + str(self.config.env_choice) + '_' + str(self.config.config) + \
					   '_seed' + str(self.seed) + \
		               ('_matd3' if self.is_matd3 else '') + \
		               ('_maddpg' if self.is_maddpg else '')



		self.save_foldername = 'R_MERL/'
		if not os.path.exists(self.save_foldername): os.makedirs(self.save_foldername)
		self.metric_save = self.save_foldername + 'metrics/'+self.config.config+'/'
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

class MERL:
	"""Policy Gradient Algorithm main object which carries out off-policy learning using policy gradient
	   Encodes all functionalities for 1. TD3 2. DDPG 3.Trust-region TD3/DDPG 4. Advantage TD3/DDPG

			Parameters:
				args (int): Parameter class with all the parameters

			"""

	def __init__(self, args):
		self.args = args

		######### Initialize the Multiagent Team of agents ########
		self.agents = Agent(self.args, id)
		self.prey_agent = PreyAgent(self.args, -1)
		self.test_agent = TestAgent(self.args, 991)

		###### Buffer and Model Bucket as references to the corresponding agent's attributes ####
		self.predator_buffer_bucket = [buffer.tuples for buffer in self.agents.buffer]
		self.prey_buffer_bucket = [buffer.tuples for buffer in self.prey_agent.buffer]

		self.popn_bucket = self.agents.popn
		self.predator_rollout_bucket = self.agents.rollout_actor
		self.prey_rollout_bucket = self.prey_agent.rollout_actor
		self.predator_test = self.test_agent.predator
		self.prey_test = self.test_agent.prey

		######### EVOLUTIONARY WORKERS ############
		if self.args.popn_size > 0:
			self.evo_task_pipes = [Pipe() for _ in range(args.popn_size * args.num_evals)]
			self.evo_result_pipes = [Pipe() for _ in range(args.popn_size * args.num_evals)]
			self.evo_workers = [Process(target=rollout_worker, args=(
				self.args, i, 'evo', self.evo_task_pipes[i][1], self.evo_result_pipes[i][0],
				self.predator_buffer_bucket, self.prey_buffer_bucket, self.popn_bucket, self.prey_rollout_bucket, True, args.config.config)) for i in
			                    range(args.popn_size * args.num_evals)]
			for worker in self.evo_workers: worker.start()

		######### POLICY GRADIENT WORKERS ############
		if self.args.rollout_size > 0:
			self.pg_task_pipes = Pipe()
			self.pg_result_pipes = Pipe()
			self.pg_workers = [
				Process(target=rollout_worker, args=(self.args, 0, 'pg', self.pg_task_pipes[1], self.pg_result_pipes[0],
				                                     self.predator_buffer_bucket, self.prey_buffer_bucket, self.predator_rollout_bucket, self.prey_rollout_bucket,
				                                     self.args.rollout_size > 0, args.config.config))]
			for worker in self.pg_workers: worker.start()

		######### TEST WORKERS ############
		self.test_task_pipes = Pipe()
		self.test_result_pipes = Pipe()
		self.test_workers = [Process(target=rollout_worker,
		                             args=(self.args, 0, 'test', self.test_task_pipes[1], self.test_result_pipes[0],
		                                   None, None, self.predator_test, self.prey_test, False, args.config.config))]
		for worker in self.test_workers: worker.start()

		#### STATS AND TRACKING WHICH ROLLOUT IS DONE ######
		self.best_score = -999;
		self.total_frames = 0;
		self.gen_frames = 0;
		self.test_trace = []

	def make_teams(self, num_agents, popn_size, num_evals):

		temp_inds = []
		for _ in range(num_evals): temp_inds += list(range(popn_size))

		all_inds = [temp_inds[:] for _ in range(num_agents)]
		for entry in all_inds: random.shuffle(entry)

		teams = [[entry[i] for entry in all_inds] for i in range(popn_size * num_evals)]

		return teams

	def train(self, gen, test_tracker, prey_tracker):
		"""Main training loop to do rollouts and run policy gradients

			Parameters:
				gen (int): Current epoch of training

			Returns:
				None
		"""

		# Test Rollout
		if gen % self.args.test_gap == 0:
			self.test_agent.make_champ_team(self.agents, self.prey_agent)  # Sync the champ policies into the TestAgent
			self.test_task_pipes[0].send("START")

		# Figure out teams for Coevolution
		teams = [[i] for i in list(range(args.popn_size))]  # Homogeneous case is just the popn as a list of lists to maintain compatibility

		########## START EVO ROLLOUT ##########
		if self.args.popn_size > 0:
			for pipe, team in zip(self.evo_task_pipes, teams):
				pipe[0].send(team)

		########## START POLICY GRADIENT ROLLOUT ##########
		if self.args.rollout_size > 0 and not RANDOM_BASELINE:
			# Synch pg_actors to its corresponding rollout_bucket
			self.agents.update_rollout_actor()
			self.prey_agent.update_rollout_actor()

			# Start rollouts using the rollout actors
			self.pg_task_pipes[0].send('START')  # Index 0 for the Rollout bucket

			############ POLICY GRADIENT UPDATES #########
			# Spin up threads for each agent
			self.agents.update_parameters()


		#PREY
		self.prey_agent.update_parameters()



		all_fits = []
		####### JOIN EVO ROLLOUTS ########
		if self.args.popn_size > 0:
			for pipe in self.evo_result_pipes:
				entry = pipe[1].recv()
				team = entry[0];
				fitness = entry[1][0]
				frames = entry[2]

				for agent_id, popn_id in enumerate(team):
					self.agents.fitnesses[popn_id].append(utils.list_mean(fitness))  ##Assign
				all_fits.append(utils.list_mean(fitness))
				self.total_frames += frames

		####### JOIN PG ROLLOUTS ########
		pg_fits = []
		if self.args.rollout_size > 0 and not RANDOM_BASELINE:
			entry = self.pg_result_pipes[1].recv()
			pg_fits = entry[1][0]
			self.total_frames += entry[2]

		####### JOIN TEST ROLLOUTS ########
		test_fits = []; prey_score = 0.0
		if gen % self.args.test_gap == 0:
			entry = self.test_result_pipes[1].recv()
			test_fits = entry[1][0]
			prey_score = mod.list_mean(entry[1][1])
			prey_tracker.update([prey_score], self.total_frames)
			test_tracker.update([mod.list_mean(test_fits)], self.total_frames)
			self.test_trace.append(mod.list_mean(test_fits))

		# Evolution Step
		self.agents.evolve()

		#Save models periodically
		if gen % 20 == 0:
			torch.save(self.test_agent.predator[0].state_dict(), self.args.model_save + 'predator_' + self.args.savetag)
			torch.save(self.test_agent.prey[0].state_dict(), self.args.model_save + 'prey_' + self.args.savetag)
			print("Models Saved")

		return all_fits, pg_fits, test_fits, prey_score


if __name__ == "__main__":
	args = Parameters()  # Create the Parameters class
	test_tracker = utils.Tracker(args.metric_save, [args.log_fname], '.csv')  # Initiate tracker
	prey_tracker = utils.Tracker(args.metric_save, ['prey_'+args.log_fname], '.csv')  # Initiate tracker
	selects_tracker = utils.Tracker(args.metric_save, ['selects_' + args.log_fname], '.csv')
	torch.manual_seed(args.seed);
	np.random.seed(args.seed);
	random.seed(args.seed)  # Seeds
	if args.config.env_choice == 'hyper': from envs.hyper.PowerPlant_env import Fast_Simulator  # Main Module needs access to this class for some reason

	# INITIALIZE THE MAIN AGENT CLASS
	ai = MERL(args)
	print('Running ', args.config.env_choice, 'with config ', args.config.config, ' Predator State_dim:', args.pred_state_dim, 'Prey_state_dim', args.prey_state_dim,
	      'Action_dim', args.action_dim)
	time_start = time.time()

	###### TRAINING LOOP ########
	for gen in range(1, 10000000000):  # RUN VIRTUALLY FOREVER

		# ONE EPOCH OF TRAINING
		popn_fits, pg_fits, test_fits, prey_score = ai.train(gen, test_tracker, prey_tracker)

		# PRINT PROGRESS
		print('Ep:/Frames', gen, '/', ai.total_frames, 'Popn stat:', mod.list_stat(popn_fits), 'PG_stat:',
		      mod.list_stat(pg_fits),
		      'Test_trace:', [pprint(i) for i in ai.test_trace[-5:]], 'FPS:',
		      pprint(ai.total_frames / (time.time() - time_start)), 'Evo', args.scheme, 'Prey Score:', prey_score)

		#Update elites tracker
		if gen >2 and args.popn_size > 0:
			#elites_tracker.update([ai.agents[0].evolver.rl_res['elites']], gen)
			selects_tracker.update([ai.agents.evolver.rl_res['selects']], gen)

		if ai.total_frames > args.frames_bound:
			break

	###Kill all processes
	try: ai.pg_task_pipes[0].send('TERMINATE')
	except: None
	try: ai.test_task_pipes[0].send('TERMINATE')
	except: None
	try:
		for p in ai.evo_task_pipes: p[0].send('TERMINATE')
	except: None
	print('Finished Running ', args.savetag)
	exit(0)
