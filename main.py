import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import numpy as np

from src import TextGenerator
from utils import Preprocessing

file = 'data/book.txt'
window = 4

class DatasetMaper(Dataset):
	'''
	Handles batches of dataset
	'''
	def __init__(self, x, y):
		self.x = x
		y = np.eye(vocab_len, dtype='float')[y]
		self.y = np.reshape(y, (y.shape[0], y.shape[2]))
		
	def __len__(self):
		return len(self.x)
		
	def __getitem__(self, idx):
		return self.x[idx], self.y[idx]


def train(sequences, targets):

	model = TextGenerator()
	
	training_set = DatasetMaper(sequences, targets)
	loader_training = DataLoader(training_set, batch_size=2)
	
	optimizer = optim.RMSprop(model.parameters(), lr=0.01)
	
	for epochs in range(1):
	
		model.train()
		
		for x_batch, y_batch in loader_training:

			x = x_batch.type(torch.LongTensor)
			y = y_batch.type(torch.FloatTensor)
		
			
			y_pred = model(x)
			
			# loss = F.binary_cross_entropy(y_pred, y)
				
			# optimizer.zero_grad()
				
			# loss.backward()
				
			# optimizer.step()
			break
	
	pass
		

if __name__ == '__main__':

	preprocessing = Preprocessing()
	text = preprocessing.read_dataset(file)
	char_to_idx, idx_to_char = preprocessing.create_dictionary(text)
	x, y = preprocessing.build_sequences_target(text, char_to_idx, window=window)

	vocab_len = len(char_to_idx)
	
	sequences = x.copy()
	targets = y.copy()
	
	train(sequences, targets)
	