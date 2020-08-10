# Text Generation with Bi-LSTMs in PyTorch

This repository aims to present a model for text generation by implementing a Bi-LSTM based model coded in PyTorch. Some ideas where taken from the articles <a href="https://arxiv.org/pdf/1908.04332.pdf">LSTM vs. GRU vs. Bidirectional RNN for script generation</a> and <a href="https://www.sciencedirect.com/science/article/pii/S1319157820303360">The survey: Text generation models in deep learning</a>.

## 1. Data
The results showed in section 5 were produced by training the model with the first chapter of the book "Jack Among the Indians" by George Bird Grinnell. Below are shown the credentials of the book as well as the first lines of the used chapeter.
Credentials: 
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
down the steps, Jack closely following. The train had almost stopped,
and Jack bent forward over the porter's head to try to see the platform
and to learn who was there to meet him. Suddenly he caught sight of
three horses grazing not far from the station, and he shouted, "Oh,
there's Pawnee! Look, Thomas! that's my riding-horse; that brown with
the saddle on."
```
## 2. The model
The book is cleaned and preprocessed under the tokens based technique. The architecture of the model is made up of a Bi-LSTM layer followed by a unidirectional LSTM as well as a linear layer. The following image describes the model architecure. 

## 3. How to use
In order to install the correct versions of each dependency, it is highly suggested to work under a virtual environment. In this case, I'm using the ``pipenv`` environment. To install the dependencies you just need type:
´´´
pipenv install
´´´
then, in order to lauch the environment you would need to type:
´´´
pipenv shell
´´´
