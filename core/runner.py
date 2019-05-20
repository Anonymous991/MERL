from core import mod_utils as utils
import numpy as np, random, sys
from envs.env_wrapper import RoverDomainPython


#Rollout evaluate an agent in a complete game
def rollout_worker(args, id, type, task_pipe, result_pipe, data_bucket, models_bucket, store_transitions, random_baseline):
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

	if type == 'test': NUM_EVALS = args.num_test
	elif type == 'pg': NUM_EVALS = args.rollout_size
	elif type == 'evo': NUM_EVALS = 10 if not args.config.env_choice == 'motivate' else 1 
	else: sys.exit('Incorrect type')

	if args.rollout_size == 0: store_transitions = False  # Evolution does not need to store data

	env = RoverDomainPython(args, NUM_EVALS)
	np.random.seed(id); random.seed(id)

	while True:
		teams_blueprint = task_pipe.recv() #Wait until a signal is received  to start rollout
		if teams_blueprint == 'TERMINATE': exit(0)  # Kill yourself

		# Get the current team actors
		if type == 'test' or type == 'pg': team = [models_bucket[0] for _ in range(args.config.num_agents)]
		elif type == "evo": team = [models_bucket[0][teams_blueprint[0]] for _ in range(args.config.num_agents)]
		else: sys.exit('Incorrect type')


		fitness = [None for _ in range(NUM_EVALS)]; frame=0
		joint_state = env.reset(); rollout_trajectory = [[] for _ in range(args.config.num_agents)]
		joint_state = utils.to_tensor(np.array(joint_state))

		while True: #unless done

			if random_baseline:
				joint_action = [np.random.random((NUM_EVALS, args.state_dim))for _ in range(args.config.num_agents)]
			elif type == 'pg':
				joint_action = [team[i][0].noisy_action(joint_state[i,:], head=i).detach().numpy() for i in range(args.config.num_agents)]
			else:
				joint_action = [team[i].clean_action(joint_state[i, :], head=i).detach().numpy() for i in range(args.config.num_agents)]

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


			#Push experiences to memory
			if store_transitions:
				if not args.is_matd3 and not args.is_maddpg: #Default for normal MERL
					for agent_id in range(args.config.num_agents):
						for universe_id in range(NUM_EVALS):
							if not done[universe_id]:
								rollout_trajectory[agent_id].append([np.expand_dims(utils.to_numpy(joint_state)[agent_id,universe_id, :], 0),
															  np.expand_dims(utils.to_numpy(next_state)[agent_id, universe_id, :], 0),
															  np.expand_dims(joint_action[agent_id,universe_id, :], 0),
															  np.expand_dims(np.array([reward[agent_id, universe_id]], dtype="float32"), 0),
															  np.expand_dims(np.array([done[universe_id]], dtype="float32"), 0),
															  universe_id,
								                              type])

				else: #FOR MATD3/MADDPG - requires global state [concatenation of all observations and actions]
					for universe_id in range(NUM_EVALS):
						if not done[universe_id]:
							rollout_trajectory[0].append(
								[np.expand_dims(utils.to_numpy(joint_state)[:, universe_id, :], 0),
								 np.expand_dims(utils.to_numpy(next_state)[:, universe_id, :], 0),
								 np.expand_dims(joint_action[:, universe_id, :], 0), #[batch, agent_id, :]
								 np.array([reward[:, universe_id]], dtype="float32"),
								 np.expand_dims(np.array([done[universe_id]], dtype="float32"), 0),
								 universe_id,
								 type])


			joint_state = next_state
			frame+=NUM_EVALS

			#DONE FLAG IS Received
			if sum(done)==len(done):
				#Push experiences to main
				if store_transitions:
					for agent_id, buffer in enumerate(data_bucket):
						for entry in rollout_trajectory[agent_id]:
							temp_global_reward = fitness[entry[5]]
							entry[5] = np.expand_dims(np.array([temp_global_reward], dtype="float32"), 0)
							buffer.append(entry)

				break



		#Send back id, fitness, total length and shaped fitness using the result pipe
		result_pipe.send([teams_blueprint, [fitness], frame])




