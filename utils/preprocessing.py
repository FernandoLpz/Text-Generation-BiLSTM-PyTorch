
class Preprocessing:
	
	@staticmethod
	def read_dataset(file):
		
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
		
		return raw_text
		
	@staticmethod
	def create_dictionary(text):
		
		dictionary = dict()
				
		
