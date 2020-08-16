# Text Generation with Bi-LSTMs in PyTorch

This repository presents a model for text generation using Bi-LSTM and LSTM recurrent neural networks. The model is implemented using PyTorch's LSTMCells. 

## 1. Files
- The ``data`` directory contains the text which we will work with. 
- The ``src`` directory contains the file ``model.py``which contains the neural net definition
- The ``utils`` directory contains helper functions such as the preprocessor and the parser
- The ``weights`` directory contains the trained weights.

## 2. The model
The architecture of the proposed neural network consists of an embedding layer followed by a Bi-LSTM as well as a LSTM layer. Right after, the latter LSTM is connected to a linear layer. The following image describes the model architecure. 
<p align="center">
<img src='img/bilstm_maths.jpg'>
</p>

## 3. Dependencies
In order to install the correct versions of each dependency, it is highly suggested to work under a virtual environment. In this case, I'm using the ``pipenv`` environment. To install the dependencies you just need type:
```
pipenv install
```
then, in order to lauch the environment you would need to type:
```
pipenv shell
```
## 4. Demo
For this demos we are going to make use of the book that is in the ``data/book`` directory, the credentials of the book are:
```
book_name: Jack Among the Indians
author: George Bird Grinnell
chapter: 1
url: https://www.gutenberg.org/cache/epub/46205/pg46205.txt
```
First lines of the book:
```
The train rushed down the hill, with a long shrieking whistle, and then
began to go more and more slowly. Thomas had brushed Jack off and
thanked him for the coin that he put in his hand, and with the bag in
one hand and the stool in the other now went out onto the platform and
down the steps, Jack closely following.
```
The best results were obtained by training the model with the following parameters:
```
python -B main.py --window 100 --epochs 50 --hidden_dim 128 --batch_size 128 --learning_rate 0.001
```
The weights of the trained model are stored in the ``weights/``directory. 
To generate text, we are going to load the weights of the trained model as follows:
```
python -B main.py --load_model True --model [your_model.pt]
```
