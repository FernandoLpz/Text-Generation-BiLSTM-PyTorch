from utils import Preprocessing

file = 'data/book.txt'

if __name__ == '__main__':
	preprocessing = Preprocessing()
	preprocessing.read_dataset(file)