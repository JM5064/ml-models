# Designing Neural Network Architectures Using Reinforcement Learning

Implementation of [Designing Neural Network Architectures Using Reinforcement Learning](https://arxiv.org/pdf/1611.02167)

<img src="https://lilianweng.github.io/posts/2020-08-06-nas/MetaQNN.png" width="600">

MetaQNN performs neural architecture search for convolutional neural networks using Q-learning with an $\varepsilon$-greedy exploration strategy.

We use the following parameters for our state space:

| Layer Type            | Layer Parameters | Parameter Values |
| --------------------- | ---------------- | ---------------- |
| **Convolution (C)** | $i$ ~ Layer depth <br> $f$ ~ Receptive field size <br> $l$ ~ Stride <br> $d$ ~ # receptive fields <br> $n$ ~ Representation size | $\le$ 8 <br> Square $\in$ {1, 3, 5} <br> Square. Always equal to 1 <br> $\in$ {64, 128, 256} <br> $\le$ 32 |
| **Pooling (P)** | $i$ ~ Layer depth <br> $(f, l)$ ~ (Receptive field size, Strides) <br> $n$ ~ Representation size | $\le$ 8 <br> Square $\in$ {(5, 3), (3, 2), (2, 2)} <br> $\le$ 32 |
| **Fully Connected (FC)** | $i$ ~ Layer depth <br> $n$ ~ # consecutive FC layers <br> $d$ ~ # neurons | $\le$ 8 <br> $\lt$ 3 <br> $\in$ {128, 256} |
| **Termination State** |  | Softmax |

The neural architecture search was performed using the CIFAR-10 dataset. Due to compute constraints, each model was trained for 12 epochs, with a total of 285 models trained overall. 

---

## Results

After performing the search, we picked the top 3 models found and trained each for 50 epochs. 
Here are the architectures for the top 3 models found: 

`C(256, 3, 1) - C(256, 5, 1) - P(2, 2) - C(256, 3, 1) - C(256, 5, 1) - P(3, 2) - C(64, 1, 1) - SM(10)`

89.3% Test Accuracy

<br>

`C(64, 3, 1) - C(128, 3, 1) - C(128, 5, 1) - C(128, 1, 1) - P(5, 3) - C(128, 5, 1) - C(128, 5, 1) - SM(10)`

88.9% Test Accuracy

<br>

`C(256, 3, 1) - C(128, 3, 1) - P(5, 3) - C(128, 1, 1) - C(128, 3, 1) - C(64, 3, 1) - P(3, 2) - SM(10)`

87.6% Test Accuracy

<br>

`C(d, f, l)` represents a Convolution layer with $d$ receptive fields, kernel size $f$, and stride $l$

`P(f, l)` represents a Pooling layer with kernel size $f$ and stride $l$

`SM(10)` represents a softmax Termination layer, producing 10 outputs.
