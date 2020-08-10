import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from src import TextGenerator
from utils import Preprocessing
from utils import parameter_parser

class Execution:

	def __init__(self, args):
		self.file = 'data/book.txt'
		self.window = args.window
		self.batch_size = args.batch_size
		self.learning_rate = args.learning_rate
		self.num_epochs = args.num_epochs
		
		self.targets = None
		self.sequences = None
		self.vocab_size = None
		self.word_to_idx = None
		self.idx_to_word = None
		
	def prepare_data(self):
	
		# Initialize preprocessor object
		preprocessing = Preprocessing()
		
		# The 'file' is loaded and split by word
		text = preprocessing.read_dataset(self.file)
		
		# Given 'text', it is created two dictionaries
		# a dictiornary about: from word to index
		# a dictorionary about: from index to word
		self.word_to_idx, self.idx_to_word = preprocessing.create_dictionary(text)
		
		# Given the 'window', it is created the set of training sentences as well as
		# the set of target words
		self.sequences, self.targets = preprocessing.build_sequences_target(text, self.word_to_idx, window=self.window)
			
		# Gets the vocabuly size
		self.vocab_size = len(self.word_to_idx)
		

	def train(self, args):
	
		# Model initialization
		model = TextGenerator(args, self.vocab_size)
		# Optimizer initialization
		optimizer = optim.RMSprop(model.parameters(), lr=self.learning_rate)
		# Defining number of batches
		num_batches = int(len(self.sequences) / self.batch_size)
		# Set model in training mode
		model.train()
		
		# Training pahse
		for epoch in range(self.num_epochs):
			# Mini batches
			for i in range(num_batches):
			
				# Batch definition
				try:
					x_batch = self.sequences[i * self.batch_size : (i + 1) * self.batch_size]
					y_batch = self.targets[i * self.batch_size : (i + 1) * self.batch_size]
				except:
					x_batch = self.sequences[i * self.batch_size :]
					y_batch = self.targets[i * self.batch_size :]
			
				# Convert numpy array into torch tensors
				x = torch.from_numpy(x_batch).type(torch.LongTensor)
				y = torch.from_numpy(y_batch).type(torch.LongTensor)
				
				# Feed the model
				y_pred = model(x)
				# Loss calculation
				loss = F.cross_entropy(y_pred, y.squeeze())
				# Clean gradients
				optimizer.zero_grad()
				# Calculate gradientes
				loss.backward()
				# Updated parameters
				optimizer.step()
			
			print("Epoch: %d,  loss: %.5f " % (epoch, loss.item()))
			
			
		torch.save(model.state_dict(), 'weights/textGenerator_words_2.pt')
	
	@staticmethod
	def generator(model, sequences, idx_to_word, n_chars):
		
		model.eval()
		
		softmax = nn.Softmax(dim=1)
		
		start = np.random.randint(0, len(sequences)-1)
		pattern = sequences[start]
		
		print("\nSeed: \n")
		print(''.join([idx_to_word[value] for value in pattern]), "\"")
		
		full_prediction = pattern.copy()
		
		for i in range(n_chars):
		
			pattern = torch.from_numpy(pattern).type(torch.LongTensor)
			pattern = pattern.view(1,-1)
			
			prediction = model(pattern)
			prediction = softmax(prediction)
			
			prediction = prediction.squeeze().detach().numpy()
			arg_max = np.argmax(prediction)
			
			pattern = pattern.squeeze().detach().numpy()
			pattern = pattern[1:]
			pattern = np.append(pattern, arg_max)

			full_prediction = np.append(full_prediction, arg_max)
			
		print("Prediction: \n")
		print(''.join([idx_to_word[value] for value in full_prediction]), "\"")

if __name__ == '__main__':
	
	args = parameter_parser()
	
	if args.load_model == True:
		if os.path.exists(args.model):
			
			model = TextGenerator(args)
			model.load_state_dict(torch.load('weights/textGenerator_words_2.pt'))
			
			execution = Execution(args)
			execution.prepare_data()
			
			sequences = execution.sequences
			idx_to_word = execution.idx_to_word
			
			execution.generator(model, sequences, idx_to_word, 500)
			
	else:
		execution = Execution(args)
		execution.prepare_data()
		execution.train(args)

		sequences = execution.sequences
		idx_to_word = execution.idx_to_word
		vocab_size = execution.vocab_size
		
		model = TextGenerator(args, vocab_size)
		model.load_state_dict(torch.load('weights/textGenerator_words_2.pt'))
		
		execution.generator(model, sequences, idx_to_word, 500)