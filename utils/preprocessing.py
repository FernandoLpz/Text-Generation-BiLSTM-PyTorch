
import numpy as np
import re
class Preprocessing:
	
	@staticmethod
	def read_dataset(file):
		
		letters = ['a','b','c','d','e','f','g','h','i','j','k','l','m',
					'n','o','p','q','r','s','t','u','v','w','x','y','z',' ']
		
		# Open raw file
		with open(file, 'r') as f:
			raw_text = f.readlines()
			
		# Transform each line into lower
		raw_text = [line.lower() for line in raw_text]
		
		# Create a string which contains the entire text
		text_string = ''
		for line in raw_text:
			text_string += line.strip()
		
		# Create an array by char
		text = list()
		for char in text_string:
			text.append(char)
	
		# Remove all symbosl and just keep letters
		text = [char for char in text if char in letters]
		
		# symbols = ['.',',','-','_','+','*','"','!','?','=','^','¨','{','}','[',']','$','%','&','/','(',')','|',';',':']
		
		# # Open raw file
		# with open(file, 'r') as f:
		# 	raw_text = f.readlines()
			
		# # Transform each line into lower
		# raw_text = [line.lower().strip() for line in raw_text]
		
		# # Create a string which contains the entire text
		# text = ''
		# for text_line in raw_text:
		# 	clean_text = ''
		# 	# Remove specific symblos
		# 	for char in text_line:
		# 		if char not in symbols:
		# 			clean_text += char
		# 	clean_text += ' '
		# 	text += clean_text
			
			

		# # Entire text split by word
		# text = text.split()
	
		return text
		
	@staticmethod
	def create_dictionary(text):
		
		word_to_idx = dict()
		idx_to_word = dict()
		
		idx = 0
		for word in text:
			if word not in word_to_idx.keys():
				word_to_idx[word] = idx
				idx_to_word[idx] = word
				idx += 1
				
		print("Vocab: ", len(word_to_idx))
		
		return word_to_idx, idx_to_word
		
	@staticmethod
	def build_sequences_target(text, word_to_idx, window):
		
		x = list()
		y = list()
	
		for i in range(len(text)):
			try:
				# Get window of words from text
				# Then, transform it into its idx representation
				sequence = text[i:i+window]
				sequence = [word_to_idx[word] for word in sequence]
				
				# Get word target
				# Then, transfrom it into its idx representation
				target = text[i+window]
				target = word_to_idx[target]
				
				# Save sequences and targets
				x.append(sequence)
				y.append(target)
			except:
				pass
		
		x = np.array(x)
		y = np.array(y)
		
		return x, y
		
