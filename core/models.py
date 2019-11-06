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




# Initialize Policy weights
def sample_weight_uniform(m):
	classname = m.__class__.__name__
	if classname.find('Linear') != -1:
		torch.nn.init.uniform_(m.weight, a=-1.0, b=1.0)
		torch.nn.init.uniform_(m.bias, a=-1.0, b=1.0)


# Initialize Policy weights
def sample_weight_normal(m):
	classname = m.__class__.__name__
	if classname.find('Linear') != -1:
		torch.nn.init.normal_(m.weight, mean=0.0, std=1.0)
		torch.nn.init.normal_(m.bias, mean=0.0, std=1.0)






