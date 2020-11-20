
from torch import nn

class BinClassifier(nn.Module):
	def __init__(self, state_dim, num_classes):
		super(BinClassifier, self).__init__()
		self.model = nn.Sequential(
			nn.Linear(state_dim, 200), nn.ReLU(), 
			nn.Linear(200, 200), nn.ReLU(), 
			nn.Linear(200, num_classes)
		)
	def forward(self, x):
		return self.model(x)
