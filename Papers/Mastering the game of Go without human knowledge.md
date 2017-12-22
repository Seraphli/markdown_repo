# Mastering the game of Go without human knowledge

## What

>   Starting tabula rasa, our new program AlphaGo Zero achieved superhuman performance, winning 100–0 against the previously published, champion-defeating AlphaGo.



## How

>   Our new method uses a deep neural network $f_{\theta}$ with parameters $\theta$. This neural network takes as an input the raw board representation s of the position and its history, and outputs both move probabilities and a value, $(\boldsymbol{p}, v)= f_{\theta}(s)$.

Each state $s$, we get $\boldsymbol{p}$ from $f_{\theta}(s)$, then combining with MCTS, we get MCTS search outputs probabilities $\pi$.

The terminal position $s_T$ is scored according to the rules of the game to compute the game winner $z$.

A game terminates at step $T$ when both players pass, when the search value drops below a resignation threshold or when the game exceeds a maximum length.

Positions in the queue are evaluated by the neural network using a mini­batch size of 8; the search thread is locked until evaluation completes.

The leaf node is expanded and each edge (sL, a) is initialized to {N(sL, a)=0, W(sL, a)=0, Q(sL, a)=0, P(sL, a)=pa} 

**Q:**

Once the search is complete, search probabilities π are returned, proportional to N1/τ, where N is the visit count of each move from the root state and τ is a parameter controlling temperature. 



### Term

**Virtual loss**. If several threads start from the root at the same time, it is possible that they traverse the tree for a large part in the same way. Simulated games might start from leaf nodes, which are in the neighborhood of each other. It can even happen that simulated games begin from the same leaf node. Because a search tree typically has millions of nodes, it may be redundant to explore a rather small part of the tree several times. Coulom suggests to assign one "virtual loss" when a node is visited by a thread (i.e., in phase 1). Hence, the value of this node will be decreased. The next thread will only select the same node if its value remains better than its siblings’ values. The virtual loss is removed when the thread that gave the virtual loss starts propagating the result of the finished simulated game (i.e., in phase 4). Owing to this mechanism, nodes that are clearly better than others will still be explored by all threads, while nodes for which the value is uncertain will not be explored by more than one thread. Hence, this method keeps a certain balance between exploration and exploitation in a parallelized MCTS program.

### Overall Method

1.  create a player model *eq1*, either from random initialized weight or trained model
2.  self-play, using MCTS to generate training data $(s, \boldsymbol{\pi}, z)$. Use *eq3* for PUCT algorithm
3.  use loss function *eq6* to optimize $\theta$

### MCTS

We get the current state $s$. Set this state as root state $s_0$.

Each Simulation we do:

1.  Select: use PUCT algorithm *eq5* to select actions, and finally reach leaf node $s_L$ at time-step $L$
2.  Expand: expand leaf node $s_L$ and 


At the end of the search, we selects a move using 

### Equation

1.  $(\boldsymbol{p}, v)= f_{\theta}(s)$
2.  $U(s, a) = c_{\rm puct} P(s, a) \frac{\sqrt{\sum_b{N(s,b)}}}{1+ N(s, a)}$
3.  $P(s,a)=(1-\varepsilon)p_a+\varepsilon\eta_a$, $\boldsymbol{\eta} \sim \rm{Dir}(0.03)$ and $\varepsilon=0.25$
4.  $W(s_t, a_t) = W(s_t, a_t) + v$
5.  $Q(s_t,a_t)=\frac{W(s_t,a_t)}{N(s_t,a_t)}$
6.  $a_t=\underset{a}{\arg\max}(Q(s_t,a)+U(s_t,a))$
7.  $l=(z-v)^2- \pi^T log\boldsymbol{p} + c\Vert\theta\Vert^2$ 
8.  $\{N(s_L,a)=0, W(s_L,a)=0,Q(s_L,a)=0,P(s_L,a)=p_a\}$
9.  $\pi(a|s_0)=N(s_0,a)$

### Summary

1.  one neural network consists of residual blocks of convolutional layers with batch normalization and rectifier nonlinearities.

### Tips

1.  `np.random.dirichlet` can be used to draw simple from Dirichlet Distribution.

## Reference



1.  [Parallel Monte-Carlo Tree Search][1]

[1]: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.159.4373&amp;rep=rep1&amp;type=pdf	"Parallel Monte-Carlo Tree Search"

