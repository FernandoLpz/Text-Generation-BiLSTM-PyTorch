import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class TextGenerator(nn.ModuleList):
	def __init__(self, args):
		super(TextGenerator, self).__init__()
		
		self.batch_size = args.batch_size
		self.hidden_dim = args.hidden_dim
		self.input_size = 27
		self.num_classes = 27
		self.sequence_len = args.window
		self.dropout = nn.Dropout(0)
		self.embedding = nn.Embedding(self.input_size, self.hidden_dim, padding_idx=0)
		self.lstm_cell_1 = nn.LSTMCell(self.hidden_dim, self.hidden_dim)
		self.lstm_cell_2 = nn.LSTMCell(self.hidden_dim, self.hidden_dim)
		self.fc_1 = nn.Linear(self.hidden_dim, self.num_classes)
		
	def forward(self, x):
	
		# batch_size x hidden_size
		hidden_state = torch.zeros(x.size(0), self.hidden_dim)
		cell_state = torch.zeros(x.size(0), self.hidden_dim)
		hidden_state_2 = torch.zeros(x.size(0), self.hidden_dim)
		cell_state_2 = torch.zeros(x.size(0), self.hidden_dim)

		# weights initialization
		torch.nn.init.xavier_normal_(hidden_state)
		torch.nn.init.xavier_normal_(cell_state)
		torch.nn.init.xavier_normal_(hidden_state_2)
		torch.nn.init.xavier_normal_(cell_state_2)
		
		# From idx to embedding
		out = self.embedding(x)
		
		# Prepare the shape for LSTMCell
		out = out.view(self.sequence_len, x.size(0), -1)
		
		# Unfolding LSTM
		# Last hidden_state will be used to feed the fully connected neural net
		for i in range(self.sequence_len):
		 	hidden_state, cell_state = self.lstm_cell_1(out[i], (hidden_state, cell_state))
		 	hidden_state = self.dropout(hidden_state)
		 	cell_state = self.dropout(cell_state)
		 	hidden_state_2, cell_state_2 = self.lstm_cell_2(hidden_state, (hidden_state_2, cell_state_2))
		 	hidden_state_2 = self.dropout(hidden_state_2)
		 	cell_state_2 = self.dropout(cell_state_2)
		 	
		# Last hidden state is passed through a fully connected neural net
		out = self.fc_1(hidden_state_2)

		
		return out
		
		