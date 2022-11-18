# Why did the distribution change?

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tiagofrepereira2012/why_did_the_distribution_change/blob/main/closed_form_simulation.ipynb#scrollTo=b1371ea7-9cea-473d-b132-a8170b8c1a3c)


In this notebook I'll try to understand this paper: http://proceedings.mlr.press/v130/budhathoki21a/budhathoki21a.pdf

The paper describe a method to detect the variables responsible for distribution changes using graphical causal models.

## Preamble

- $G$ is a graphical causal model with variables $X_1, X_2, ..., X_n$. $C=\langle G, P_X \rangle$ 
- $G$ can be factorized with a joint probability $P_{X_1,...X_n} = \prod\limits_{j=1}^{n}P_{X_j}|PA_j$, where $P_{X_j}|PA_j$ denotes the causal mechanism of $X_j$ given its direct parents $PA_j$.
- $X_k = [X_1, ... X_n]$
 
## Research Question

The research question is, why $\mathbb{E}(y|X_k) - \mathbb{E}(y|\tilde{X_k}) >> 0$. Which variables on $X_k$ were responsible for such a change in these marginal distributions?

## Hypothesis


Given two graphical causal models $C=\langle G, P_X \rangle$ and $\tilde{C}=\langle G, \tilde{P_X} \rangle$ corresponding to models that we want to measure possible changes on $P(y|X_k)$, those changes can be measured via the KL divergence pairing node by node on $C$ and $\tilde{C}$. Hence, $D(\tilde{P_X} || P_X) = \sum\limits_{k=1}^n D(\tilde{P_{X_j}}|PA_j || P_{X_j}|PA_j)$ (the KL divergence of the joint distribution sum up to the divergence of the marginals). Then, we can look individually at the marginal differences and see the most important feature.

Then, **and this is an important one**, it is hypothesized that the Shapley values of all variables will sum up to the KL divergence of the marginals from $\tilde{P_X}$ to $P_X$. So, $\sum\limits_{j=1}^n \phi_j(D) = D(\tilde{P_{X_j}} || P_{X_j})$.

## Algorithm

Their algorithm is split in two parts. First part is the **detection of a changing mechanism** and the second part is the **attribuition mechanism**.

### Detection mechanism

It consists in detecting if $\tilde{P_{X_j}}$ is different of $P_{X_j}$ given its parents. This can be measured by a conditional dependence test. In this case, imported from this paper here (https://arxiv.org/abs/1202.3775).
The tests consists in analyse if $X'_k \perp  A | PA_k$, where $X'_k = [X_k, \tilde{X_k}]$ and $A=[-1,-1,...,+1,+1]$ (A vector with the size of $X'_k$ where $a_i=-1$ if $a_i\in X_k$ and $+1$ otherwise).

Every node in both graphs, $C$ and $\tilde{C}$, is fit with an arbitrary model $\Phi$ (linear regression or some generative model).

If $X'_k \perp  A | PA_k$, holds only one model is trained $\Phi(PA_k, [X_k, \tilde{X_K}])$. If not, two models are trained $\Phi(PA_k, X_k)$ and $\Phi(PA_k, \tilde{X_k})$. 
Those models, are used to sample from the distribution to compute the shapley.

## Attributition

The attribution corresponds to the computation of the Shapley values.
Shapley has its particular algorithm that I'll not not explain here. In short, you try to evaluated the importance of each feature, by adding them one by one during the classification. The order matters, so, they run the shappley algorithm several times, permutating the features.
For every permutation, $\sum\limits_{j=1}^n \phi_j(\mathbb{E})=\mathbb{E}_{X_k \sim \tilde{P_{X_k}}}[X_k] - \mathbb{E}_{X_k \sim P_{X_k}}[X_k]$ is evaluated. $X_k$ is sampled from $\Phi$. If the permutation contains the variable $k$, $Phi$ is taken the new distribution, if not, $Phi$ is taken from the current one.

This algorithm is executed several times and the average shapley is provided as output.

This notebooks ilustrates the example from section 7.1 of the paper.
