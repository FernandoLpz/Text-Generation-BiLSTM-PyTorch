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


class DatasetMaper(Dataset):
	'''
	Handles batches of dataset
	'''
	def __init__(self, x, y):
		self.x = x
		self.y = y
		
	def __len__(self):
		return len(self.x)
		
	def __getitem__(self, idx):
		return self.x[idx], self.y[idx]

class Execution:

	def __init__(self, args):
		self.file = 'data/book.txt'
		self.window = args.window
		self.batch_size = args.batch_size
		self.learning_rate = args.learning_rate
		self.num_epochs = args.num_epochs
		
		self.sequences = None
		self.targets = None
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
	
		model = TextGenerator(args)
	
		training_set = DatasetMaper(self.sequences, self.targets)
		loader_training = DataLoader(training_set, batch_size=self.batch_size)
		
		optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
		
		for epoch in range(self.num_epochs):
		
			predictions = []
			
			model.train()
			
			for x_batch, y_batch in loader_training:
	
				x = x_batch.type(torch.LongTensor)
				y = y_batch.type(torch.LongTensor)
			
				y_pred = model(x)
				
				loss = F.cross_entropy(y_pred, y.squeeze())
					
				optimizer.zero_grad()
					
				loss.backward()
					
				optimizer.step()
				
				y_pred_idx = y_pred.squeeze().detach().numpy()
				predictions += list(np.argmax(y_pred_idx, axis=1))
				
			accuracy = self.calculate_accuracy(self.targets, predictions)
			
			print("Epoch: %d,  loss: %.5f, accuracy: %.5f " % (epoch, loss.item(), accuracy))
			
		torch.save(model.state_dict(), 'weights/textGenerator_words_2.pt')
				
	@staticmethod
	def calculate_accuracy(y_true, y_pred):
		tp = 0
		for true, pred in zip(y_true, y_pred):
			if true == pred:
				tp += 1
				
		return tp / len(y_true)
	
	@staticmethod
	def generator(model, sequences, idx_to_word, generator):
		
		model.eval()
		
		softmax = nn.Softmax(dim=1)
		
		start = np.random.randint(0, len(sequences)-1)
		pattern = sequences[start]
		
		print("Seed: \n")
		print(' '.join([idx_to_word[value] for value in pattern]), "\"")
		
		full_prediction = pattern.copy()
		
		for i in range(generator):
		
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
			
		print("Prediction: ")
		print("\"", ' '.join([idx_to_word[value] for value in full_prediction]), "\"")

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
			
			execution.generator(model, sequences, idx_to_word, 100)
			
	else:
		execution = Execution(args)
		execution.prepare_data()
		execution.train(args)

		sequences = execution.sequences
		idx_to_word = execution.idx_to_word
		
		model = TextGenerator(args)
		model.load_state_dict(torch.load('weights/textGenerator_words_2.pt'))
		
		execution.generator(model, sequences, idx_to_word, 100)