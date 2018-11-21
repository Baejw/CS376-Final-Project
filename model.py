import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(39)
torch.cuda.manual_seed_all(39)

class Model(nn.Module):
	def __init__(self):
		super(Model, self).__init__()
		n_input = 15
		n_hidden = 32
		self.bn1 = nn.BatchNorm1d(n_input, track_running_stats=False)
		self.fc1 = nn.Linear(n_input, n_hidden)
		nn.init.xavier_normal_(self.fc1.weight)
		self.bn2 = nn.BatchNorm1d(n_hidden, track_running_stats=False)
		self.fc2 = nn.Linear(n_hidden, 16)
		nn.init.xavier_normal_(self.fc2.weight)
		self.bn3 = nn.BatchNorm1d(16, track_running_stats=False)
		self.fc3 = nn.Linear(16, 1)
		nn.init.xavier_normal_(self.fc3.weight)

	def forward(self, x):
		x = self.bn1(x)
		x = F.leaky_relu(self.fc1(x))
		x = F.dropout(x, p=0.5, training=self.training)
		x = self.bn2(x)
		x_ = F.leaky_relu(self.fc2(x))
		x = F.dropout(x_, p=0.5, training=self.training)
		x = self.bn3(x)
		x = self.fc3(x)
		return x, x_