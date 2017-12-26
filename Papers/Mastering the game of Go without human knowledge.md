# Mastering the game of Go without human knowledge

## What

>   Starting tabula rasa, our new program AlphaGo Zero achieved superhuman performance, winning 100–0 against the previously published, champion-defeating AlphaGo.

## How

>   Our new method uses a deep neural network $f_{\theta}$ with parameters $\theta$. This neural network takes as an input the raw board representation s of the position and its history, and outputs both move probabilities and a value, $(\boldsymbol{p}, v)= f_{\theta}(s)$.

Each state $s$, we get $\boldsymbol{p}$ from $f_{\theta}(s)$, then combining with MCTS, we get MCTS search outputs probabilities $\pi$.

### Term

**Virtual loss**. If several threads start from the root at the same time, it is possible that they traverse the tree for a large part in the same way. Simulated games might start from leaf nodes, which are in the neighborhood of each other. It can even happen that simulated games begin from the same leaf node. Because a search tree typically has millions of nodes, it may be redundant to explore a rather small part of the tree several times. Coulom suggests to assign one "virtual loss" when a node is visited by a thread (i.e., in phase 1). Hence, the value of this node will be decreased. The next thread will only select the same node if its value remains better than its siblings’ values. The virtual loss is removed when the thread that gave the virtual loss starts propagating the result of the finished simulated game (i.e., in phase 4). Owing to this mechanism, nodes that are clearly better than others will still be explored by all threads, while nodes for which the value is uncertain will not be explored by more than one thread. Hence, this method keeps a certain balance between exploration and exploitation in a parallelized MCTS program.



**Leaf node.** Leaf node refer to node that has been added into search tree. While on the search, a node can be added to the tree only if it has been visited. Then a new node is initialized.

### Overall Method

1.  Create a player model *eq1*. The model is initialized either from random weight or trained model.
2.  Self-play. Use MCTS to play game. Each time step $t$ the game generates partial training data $(s_t, \boldsymbol{\pi_t})$. Then at the end of the game, the game is then scored to give a final reward of $r_T \in \{-1,+1\}$. Then we wrap the training data as $(s_t, \boldsymbol{\pi_t}, z_t)$, where $z_t = \pm r_T$ is the game winner from the perspective of the current player at step $t$.
3.  Sample uniformly from training data and use loss function *eq8* to optimize $\theta$.

### MCTS

We get the current state $s$. Set this state as root state $s_0$.

Each Simulation we do:

1.  Select. Use PUCT algorithm *eq7* to select actions, and finally reach leaf node $s_L$ at time-step $L$.
2.  Expand. Expand leaf node $s_L$, which is, select one child node, initialize the node using *eq9*.
3.  Evaluation. The leaf node $s_L$ is added to a queue for neural network evaluation, $(d_i(\boldsymbol{p}), v)= f_{\theta}(d_i(s_L))$.
4.  Backup. The edge statistics are updated in a backward pass through each step $t \leq L$, using *eq4*, *eq5* and *eq6*.


At the end of the search, we select a move using *eq10*.

### Equation

1.  $(\boldsymbol{p}, v)= f_{\theta}(s)$
2.  $U(s, a) = c_{\rm puct} P(s, a) \frac{\sqrt{\sum_b{N(s,b)}}}{1+ N(s, a)}$
3.  $P(s,a)=(1-\varepsilon)p_a+\varepsilon\eta_a$, $\boldsymbol{\eta} \sim \rm{Dir}(0.03)$ and $\varepsilon=0.25$
4.  $N(s_t, a_t)=N(s_t, a_t)+1$
5.  $W(s_t, a_t) = W(s_t, a_t) + v$
6.  $Q(s_t,a_t)=\frac{W(s_t,a_t)}{N(s_t,a_t)}$
7.  $a_t=\underset{a}{\arg\max}(Q(s_t,a)+U(s_t,a))$
8.  $l=(z-v)^2- \pi^T log\boldsymbol{p} + c\Vert\theta\Vert^2$ 
9.  $\{N(s_L,a)=0, W(s_L,a)=0,Q(s_L,a)=0,P(s_L,a)=p_a\}$
10.  $\pi(a|s_0)=N(s_0,a)^{1/\tau}/\sum_b N(s_0,b)^{1/\tau}$

### Parameter

1.  Softmax temperature. For the first 30 moves of each game, $\tau = 1$. For the remainder of the game, an infinitesimal temperature is used, $\tau \to 0$.
2.  Virtual loss $n_{\rm vl} = 3$
3.  Expansion threshold $n_{\rm thr} = 40$
4.  Exploration constant $c_{\rm puct} = 5$

### Tips

1.  `np.random.dirichlet` can be used to draw simple from Dirichlet Distribution.
2.  There is no pooling layer in the neural network. One neural network consists of residual blocks of convolutional layers with batch normalization and rectifier nonlinearities. There are two output from the network.
3.  The subtree below this child is retained along with all its statistics, while the remainder of the tree is discarded. 

## Reference

1.  Parallel Monte-Carlo Tree Search
2.  Mastering the game of Go without human knowledge 
3.  Mastering the game of Go with deep neural networks and tree search

