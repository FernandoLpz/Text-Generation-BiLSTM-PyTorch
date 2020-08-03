import numpy as np

from preprocessing import Preprocessing
	
if __name__ == '__main__':
	
	symbols = ['.',',','-','_','+','*','"',"'",'!','?','=','^','¨','{','}','[',']','$','%','&','/','(',')','|',';',':']
	
	file = '../data/book.txt'
	
	# Open raw file
	with open(file, 'r') as f:
		raw_text = f.readlines()
		
	# Transform each line into lower
	raw_text = [line.lower().strip() for line in raw_text]
	
	# Create a string which contains the entire text
	text_string = ''
	for text_line in raw_text:
		clean_text = ''
		for char in text_line:
			if char not in symbols:
				clean_text += char
		text_string += clean_text
		
	unique = list()
	for word in text_string.split():
		if word not in unique:
			unique.append(word)
	
	# preprocessing = Preprocessing()
	# text = preprocessing.read_dataset(file)
	
	# char_to_idx, idx_to_char = preprocessing.create_dictionary(text)
	# sequences, targets = preprocessing.build_sequences_target(text, char_to_idx, window=window)
	# vocab_size = len(char_to_idx)