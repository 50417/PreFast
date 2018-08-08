# PreFast 

A simple technique to speed up the neural net training by progressively freezing from both forward and backward pass. The technique is extension of FreezeOut[1].
![FreezeOut](https://i.imgur.com/FhdOjZT.png 'Freezeout Technique')
![PreFast](https://i.imgur.com/ihglNNe.png 'PreFast Technique')

This repository contains code for PreFast.

FreezeOut directly accelerates training by annealing layer-wise learning rates to zero on a set schedule, and excluding layers from the backward pass once their learning rate bottoms out. On the other hand, PreFast precomputes the frozen layers output when the learning rate anneals to zero and stores it to memory to exclude it from both forward and backward pass. 

## Installation
To run this script, you will need [PyTorch](http://pytorch.org) and a CUDA-capable GPU. If you wish to run it on CPU, just remove all the .cuda() calls.

## Running
To run with default parameters,

```sh
python train.py
```

Support for command line arguments can be found in [1]

[1]https://github.com/ajbrock/FreezeOut