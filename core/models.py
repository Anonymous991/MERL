import torch
import torch.nn as nn


class MultiHeadActor(nn.Module):
	"""Actor model

		Parameters:
			  args (object): Parameter class
	"""

	def __init__(self, num_inputs, num_actions, hidden_size, num_heads):
		super(MultiHeadActor, self).__init__()

		self.num_heads = num_heads
		self.num_actions = num_actions

		#Trunk
		self.linear1 = nn.Linear(num_inputs, hidden_size)
		self.linear2 = nn.Linear(hidden_size, hidden_size)

		#Heads
		self.mean = nn.Linear(hidden_size, num_actions*num_heads)
		self.noise = torch.Tensor(num_actions*num_heads)

		self.apply(weights_init_policy_fn)




	def clean_action(self, state, head=-1):
		"""Method to forward propagate through the actor's graph

			Parameters:
				  input (tensor): states

			Returns:
				  action (tensor): actions


		"""

		x = torch.tanh(self.linear1(state))
		x = torch.tanh(self.linear2(x))
		mean = torch.tanh(self.mean(x))

		if head == -1:
			return mean
		else:
			start = head*self.num_actions
			return mean[:,start:start+self.num_actions]



	def noisy_action(self, state, head=-1):

		x = torch.tanh(self.linear1(state))
		x = torch.tanh(self.linear2(x))
		mean = torch.tanh(self.mean(x))

		action = mean + self.noise.normal_(0., std=0.4)
		if head == -1:
			return action
		else:
			start = head * self.num_actions
			return action[:, start:start + self.num_actions]




	def get_norm_stats(self):
		minimum = min([torch.min(param).item() for param in self.parameters()])
		maximum = max([torch.max(param).item() for param in self.parameters()])
		means = [torch.mean(torch.abs(param)).item() for param in self.parameters()]
		mean = sum(means)/len(means)

		return minimum, maximum, mean


class QNetwork(nn.Module):
	def __init__(self, num_inputs, num_actions, hidden_size):
		super(QNetwork, self).__init__()

		# Q1 architecture
		self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
		self.linear2 = nn.Linear(hidden_size, hidden_size)
		self.linear3 = nn.Linear(hidden_size, 1)

		# Q2 architecture
		self.linear4 = nn.Linear(num_inputs + num_actions, hidden_size)
		self.linear5 = nn.Linear(hidden_size, hidden_size)
		self.linear6 = nn.Linear(hidden_size, 1)

		self.apply(weights_init_value_fn)

	def forward(self, state, action):
		x1 = torch.cat([state, action], 1)
		x1 = torch.tanh(self.linear1(x1))
		x1 = torch.tanh(self.linear2(x1))
		x1 = self.linear3(x1)

		x2 = torch.cat([state, action], 1)
		x2 = torch.tanh(self.linear4(x2))
		x2 = torch.tanh(self.linear5(x2))
		x2 = self.linear6(x2)

		return x1, x2


# Initialize Policy weights
def weights_init_policy_fn(m):
	classname = m.__class__.__name__
	if classname.find('Linear') != -1:
		torch.nn.init.xavier_uniform_(m.weight, gain=0.5)
		torch.nn.init.constant_(m.bias, 0)

# Initialize Value Fn weights
def weights_init_value_fn(m):
	classname = m.__class__.__name__
	if classname.find('Linear') != -1:
		torch.nn.init.xavier_uniform_(m.weight, gain=1)
		torch.nn.init.constant_(m.bias, 0)





