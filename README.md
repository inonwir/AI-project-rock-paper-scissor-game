# AI-project-rock-paper-scissor-game
AI/Machine Learning for image classification, implemented as rock/paper/scissor hand-game

# Introduction
We got our main influence from the game rock,paper,scissors.We thought of creating a simple command line where we use the picture we took of our hand to play with the computer by feeding it into the command line and see who wins the turn.For training the system we took at least 1000+ pictures of the 3 gestures and then we use the system to train these picture with all angle possible for example upside down,tilt 90 degree,tilt 180 degree et to generate more training sample.

# AI technique 
The main technique used to design this project is convolutional neural network.Convolutional Neural Network is a class of profound neural system that is utilized for Computer Vision or examining visual symbolism.The Convolutional Layer makes utilization of an arrangement of learnable channels.A channel is utilized to identify the nearness of particular highlights or examples in the original picture.

# System Design
The process of building a convolutional neural network consist of the following four step:<br/>
   1. Convolution
   2. Pooling
   3. Weights
   4. Full connection <br/>
          - Convolutional layers apply a convolution activity to the information, passing the outcome to the next layer. <br/>
          - Convolutional systems may incorporate nearby or worldwide pooling layers, which join the yields of neuron groups at one layer into a solitary neuron in the prior layer.<br/>
          - Fully connected layers connect every neuron in one layer to every neuron in another layer. It is in principle the same as the traditional multi-layer perceptron neural   network.<br/>
           - CNNs share weights in convolutional layers, which means that the same filter is used for each receptive field in the layer; this reduces memory footprint and improves performance.<br/>
