import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class TextGenerator(nn.ModuleList):
	def __init__(self, args, vocab_size):
		super(TextGenerator, self).__init__()
		
		self.batch_size = args.batch_size
		self.hidden_dim = args.hidden_dim
		
		self.input_size = vocab_size
		self.num_classes = vocab_size

		self.sequence_len = args.window
		
		self.dropout = nn.Dropout(0.0)
		self.embedding = nn.Embedding(self.input_size, self.hidden_dim, padding_idx=0)
		self.lstm_cell_forward = nn.LSTMCell(self.hidden_dim, self.hidden_dim)
		self.lstm_cell_backward = nn.LSTMCell(self.hidden_dim, self.hidden_dim)
		self.lstm_cell = nn.LSTMCell(self.hidden_dim * 2, self.hidden_dim * 2)
		
		self.linear = nn.Linear(self.hidden_dim * 2, self.num_classes)
		
	def forward(self, x):
	
		# batch_size x hidden_size
		hs_forward = torch.zeros(x.size(0), self.hidden_dim)
		cs_forward = torch.zeros(x.size(0), self.hidden_dim)
		hs_backward = torch.zeros(x.size(0), self.hidden_dim)
		cs_backward = torch.zeros(x.size(0), self.hidden_dim)
		hs_lstm = torch.zeros(x.size(0), self.hidden_dim * 2)
		cs_lstm = torch.zeros(x.size(0), self.hidden_dim * 2)

		# weights initialization
		torch.nn.init.kaiming_normal_(hs_forward)
		torch.nn.init.kaiming_normal_(cs_forward)
		torch.nn.init.kaiming_normal_(hs_backward)
		torch.nn.init.kaiming_normal_(cs_backward)
		torch.nn.init.kaiming_normal_(hs_lstm)
		torch.nn.init.kaiming_normal_(cs_lstm)

		# From idx to embedding
		out = self.embedding(x)
		
		# Prepare the shape for LSTM Cells
		out = out.view(self.sequence_len, x.size(0), -1)
		
		forward = []
		backward = []
		
		# Unfolding Bi-LSTM
		# Forward
		for i in range(self.sequence_len):
		 	hs_forward, cs_forward = self.lstm_cell_forward(out[i], (hs_forward, cs_forward))
		 	forward.append(hs_forward)
		 	
		# Backward
		for i in reversed(range(self.sequence_len)):
		 	hs_backward, cs_backward = self.lstm_cell_backward(out[i], (hs_backward, cs_backward))
		 	backward.append(hs_backward)
		 	
		# LSTM
		for fwd, bwd in zip(forward, backward):
			input_tensor = torch.cat((fwd, bwd), 1)
			hs_lstm, cs_lstm = self.lstm_cell(input_tensor, (hs_lstm, cs_lstm))
		 	
		# Forward + Backward
		# last_hidden_state = torch.cat((hs_forward, hs_backward), 1)

		# Last hidden state is passed through a linear layer
		out = self.linear(hs_lstm)

		return out
		