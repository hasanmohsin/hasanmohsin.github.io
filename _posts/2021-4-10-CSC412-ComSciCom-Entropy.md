---
layout: post
title: Trying to Understand Entropy
---

This post is going to try to answer the question: "What exactly is entropy"? 
Mathematically, the question is answered easily enough. Suppose we have some probability distribution $p(x)$ over a state space with $m$ outcomes $\\{x_1, x_2, ... x_m\\}$. In this post we will stick with a discrete state space, since the intuition is more clear and there are some complications with continuous ones. The entropy $$H$$ is defined for such a distribution as:

$$\begin{equation}
H[p] = - \sum^m_{i=1} p(x_i) \log p(x_i)
\end{equation}$$

Ok, but what actually *is* the entropy? How can we understand it? It is the cornerstone of Information theory, is a fundamental quantity in statistical physics, and appears all over the place in Machine Learning, so the task seems worthwhile. 

Generally it is described as some sort of measure of uncertainty, information, or surprise associated with a random event. It is also usually described as characterizing the width of a probability distribution. While these interpretations are fair, they are a little imprecise. For instance, why do we need entropy to describe "uncertainty" in the random outcome when something like the variance can describe this perfectly well? It seems like we could come up with many *ad-hoc* formulas for something like uncertainty, and these arguments don't really convince me of why entropy is special. Instead, I will present what I think is the most intuitively satisfying picture of entropy: the so-called "Wallis" derivation. I will first discuss derive the formula for entropy, and then reason through its appearance and utility in different areas of math and science. 

Unless otherwise noted, in the discussion below, I will take $\log$ to denote the logarithm with base $e$.

## Coming up with Entropy 

Suppose you have some setup for performing random experiments, and each time you run an experiment you get something out of a set of $m$ outcomes $\\{ x_1, ..., x_m \\}$. This can correspond to a coin toss with $m=2$, a roll of a single die with $m=6$ or many other scenarios. We will stick with the general setup. Now consider the following problem: how do you go about assigning probabilities, $p(x_i)$ to each outcome, **if you can't actually run the experiments**, and if you know nothing else about the system? 

### The Principle of Indifference

An intuitive solution to the problem is that if we don't know anything at all about our experimental outcome, then we have no reason to assume one outcome, $x_i$ is any more or less likely than any other, $x_j$. Thus, the most reasonable thing to do is to be *indifferent* to what each outcome actually is, and assign an equal probability to everything. This rule, intuitively understood as far back as the 1600s, is called the "principle of indifference".

### Generalizing the Principle

Let's apply the principle in a different, slightly more general way. Say we ran the experiment $N$ times, where $N$ is large. We'd get some sequence of $N$ outcomes like $(x_{m-2}, x_1, x_1, x_3, x_m,...)$, where some outcomes will repeat. Given such a sequence, we can count the number of times outcome $x_i$ occurs, and call it $n_i$. Now, if we ran the experiment $N$ times, and we saw outcome $x_i$ occur $n_i$ times, then we can estimate the probability $p(x_i) \approx \frac{n_i}{N}$, We would expect that the larger $N$ is, the closer we'll get to the true probability. If we had observed some other sequence of $N$ outomces, our probability estimates will be slightly different. In this way, the observed sequence induces an assignment of probabilities.

Now keep in mind the crux of the problem: we can't actually run these experiments, and we'll need to reason about these sequences in a different way. In fact, we'll appeal to the principle of indifference **on these sequences**. Since we don't have any other information, we'll say that all sequences are as likely to occur as each other. 

What does this mean for the probability assignments? It is possible for many sequences to give us the same "outcome counts" $\\{ n_1, n_2, ... n_m\\}$, and therefore the same assignment of probabilities. Let's denote an assignment of probabilities by the vector $\mathbf{p} = (p(x_1), p(x_2), ... p(x_m)) = (n_1/N, n_2/N, ... n_m/N)$. What we are interested in is the **probability** distribution over the possible assignments $\mathbf{p}$, which we'll denote $P(\mathbf{p})$. The principle of indifference tells us that each sequence is equally probable. This means that an assignment $p$ is more probable if there are more distinct sequences which give rise to the same assignment, or equivalently, the same set of outcome counts $\\{n_1, n_2,... n_m\\}$. In other words, if we denote by $W(\mathbf{p})$ the number of distinct sequences which have the specified probability assignments $p$, we have:

$$\begin{align} 
P(\mathbf{p}) &\propto W(\mathbf{p}) \\
&= \frac{W(\mathbf{p})}{Z}
\end{align}$$

Where $Z$ is some normalizing factor.

To calculate $W(\mathbf{p})$, we can do some combinatorics. We are interested in the number of ways we can choose $n_1$ objects of one type, $n_2$ of another type, ..., and $n_m$ of a last type, out of a total group of $N$ items. Consider a certain specific sequence which has the prescribed outcome counts. There are $N!$ possible permutations of this sequence. If $x_1$ occurs $n_1$ times, then there are $n_1!$ possible ways we can shuffle it around while giving the same sequence. Shuffling around within each outcome gives the same sequence, so the $N!$ overcounts by a factor of $n_1! n_2! n_3! ... n_m!$. Correcting for this gives us $W(\mathbf{p})$: 

$$\begin{align} W(\mathbf{p}) &= \frac{N!}{n_1! n_2!...n_m!}
\end{align}$$

Now that we have a complete description of the probability, we can ask for the **most likely** assignment $\mathbf{p^{\*}} = \mathrm{argmax} \; P(\mathbf{p})$. We will pick this $\mathbf{p^{\*}}$ as the outcome probabilities.

We can see that this involves maximizing $W(\mathbf{p})$. Since $\log$ is a monotonically increasing function, we can instead maximize the quantity $\log W(\mathbf{p})$.

$$\begin{align}
\log (W(\mathbf{p})) &= \log\Big( \frac{N!}{n_1! n_2!...n_m!} \Big) \\
&= \log(N!) - \sum_{i=1}^{m} \log(n_i!) 
\end{align}$$

Next, we take $N$ large (and consequently $n_i$ large), and apply Stirling's approximation for the factorial: $\log(n!) \approx n\log(n) - n$.

$$\begin{align}
\log (W(\mathbf{p})) &\approx N\log N - N - \sum{i=1}^{m} (n_i \log n_i - n_i) \\
&= \sum_{i=1}^{m} (n_i\log N - n_i) - \sum_{i=1}^{m} (n_i \log n_i - n_i) \\
&= \sum_{i=1}^{m} (n_i \log N - n_i \log n_i) \\
&= -N\sum_{i=1}^{m} \frac{n_i}{N} \log (\frac{n_i}{N}) \\
&= -N \sum_{i=1}^{m} p(x_i) \log p(x_i) \\
&= N H[ \mathbf{p} ]
\end{align}$$

This derivation leads us to the entropy: $H[\mathbf{p}] = \frac{1}{N} \log (W(\mathbf{p}))$. The entropy of a distribution is a quantity which reflects how likely that distribution is, under the principle of indifference.

We determine our probability $\mathbf{p}^{\*}$ by maximizing the entropy (since it is equivalent to maximizing $\log W(\mathbf{p})$, under the constraint that $\mathbf{p}$ be normalized, ie. $\sum_{i=1}^m p(x_i) = 1$. We can do this by simply setting the gradient $\nabla_\mathbf{p} H = 0$, since the function $H$ is convex. Each $p(x_k)$ is then determined as:

$$begin{align}
\frac{\partial}{\partial p(x_k)} H &= 0 \\
-\frac{\partial}{\partial p(x_i)} \sum_{i=1}^{m} p(x_i) \log p(x_i) &= 0 
\end{align}$$

$$begin{align}
\frac{\partial}{\partial p(x_k)} H &= 0 \\
-\frac{\partial}{\partial p(x_i)} \sum_{i=1}^{m} p(x_i) \log p(x_i) &= 0 \\
-\frac{\partial}{\partial p(x_k)} \sum_{i=1}^{m} p(x_i) \log p(x_i) &= 0\\
-\log p(x_k) - \frac{p(x_k)}{p(x_k)} &= 0 \\
p(x_k) &= \exp(-1) \\
p(x_k) &= \mathrm{constant}
\end{align}$$

We get the same number for each $p(x_i)$. Next we must enforce the constraint. Since the constraint is simple (and doesn't influence the optimization), we can do it manually. Upon normalization, we get a uniform distribution over the $m$ outcomes, so that $p(x_i) = \frac{1}{m}$. 

This is exactly the same result the regular principle of indifference gives us. This makes sense, we'd have cause for concern if we got any other result! 

![]({{ site.baseurl }}/images/entropy_post_figure_1.png)

But in this case what was the point of doing this long calculation? Both methods allowed us to pick the most "reasonable" probability assignments when we lack information. The difference is that by phrasing the problem as maximizing entropy, we can easily work with constraints.

For instance, suppose that in the above problem, we were given one extra piece of information: the mean outcome $\mathbb{E}[x] = \bar{x}$. This means that the probability assignments $\mathbf{p}$ must satisfy $\sum_{i=1}^{m} x_i p(x_i) = \bar{x}$. Now we must pick the probability assignment $\mathbf{p}$, under this constraint, which is most likely.

There isn't a straightforward way of doing this if we were using the regular principle of indifference, but using the entropy, it is easy. We add in the constraint as Lagrange multiplier and maximize the objective $H[\mathbf{p}] + \lambda (\bar{x} - \sum_{i=1}^{m} x_i p(x_i))$. Note that we had no need of adding a similar constraint multiplier for the normalization constraint, since it is much simpler. 

We can again solve this problem by setting the gradient to 0 with respect to $\mathbf{p}$ and $\lambda$. We get as our solution the distribution: $p(x_i) = \exp(-\lambda x_i)/Z$. This is the maximum entropy (and therefore the *most likely*) distribution with the given mean $\bar{x}$.

## Why does it show up in Physics

The path that we took to derive the entropy was first discovered in statistical mechanics. The problem of interest there was the following: we have a system of many, small, interacting particles, and we can only measure the mean total energy $\bar{E}$. We are interested in writing down a probability distribution over the possible energies for each particle, $E_i$. Trying to estimate this distribution by observing the state of any particle is basically impossible, since we can't make measurements precisely for such a small object. 

Thus, we must come up with the most "reasonable" distribution over the set $\\{E_1, E_2, ... ,E_m\\}$, under the constraint that the average energy be $\bar{E}$. This exactly mirrors the last example we calculated, so that we know the solution must be  $p(x_i) = \exp(-\lambda E_i)/Z$. Indeed, this distribution is named the "Gibbs distribution" (as well as the "softmax distribution). The entropy of this distribution is equal (within a conversion of units) to the thermodynamic entropy, usually denoted $S$.

We obtained this distribution because we thought it was the most likely, but this guess works out empirically. For instance, this, combined with other definitions of physics, such as pressure and volume, lets us derive the ideal gas law, which we know to be true. 

The interpretation of entropy as the number of ways a probability distribution can occur is also fruitful in describing the 2nd law of thermodynamics. The 2nd law is fairly complicated in general, but a special case states that as a system evolves in time, its entropy tends to increase. Using our notion of entropy, we can understand why this must be: the 


## Why does it show up in Information theory

Information theory is concerned with efficient communication. Suppose we wish to send a message, which we'll model as the output of some random source. The source outputs some character from the alphabet $\\{ x_1, x_2,.. x_m \\}$ at random, with probability $p(x_i)$. 

Here, Shannon made the key observation that, as far as efficiency is concerned, communication is related to choosing between a set of possible messages, rather than the content of those messages. If our source generates messages of $N$ characters, then to "communicate a message" means specifying one out of all possible messages. If $N$ is large then every possible message must respect the probability distribution of the source. How many such messages are there? We already made this calculation in a section above: there are $W(\mathbf{p})$ such messages. 

To communicate these messages, we simply label each one with an index $i$, and send the associated the number. We assume that at the other end, the receiver knows which message corresponds to which index. If we represent each index $i$ as a binary number, how many bits will we need to send? Each possible message must get a unique index, so that the index number must cover $W(\mathbf{p})$ possibilities. A binary number requires $\log_2 W(\mathbf{p})$ bits to represent $W(\mathbf{p})$ possibilities. 

Thus communicating one message of length $N$ requires $\log_2 W(\mathbf{p})$ bits. The average number of bits we must send per character, $L$, is:

$$\begin{align}
L &= \frac{1}{N} \log_2 W(\mathbf{p}) \\
&= \frac{1}{\log 2} \frac{1}{N}  \log W(\mathbf{p}) \\
&= \frac{1}{\log 2} H[\mathbf{p}] \\
&= H_2 [\mathbf{p}]
\end{align}$$

Where again, we see that the entropy naturally arises. We have an extra factor of $\frac{1}{\log 2}$, but we can interpret this as the unit conversion factor to bits, where we label the entropy as $H_2 [\mathbf{p}]$. Indeed some authors choose to work without the conversion, and say that the entropy is in units of "nats" rather than "bits". 

Thus, this encoding scheme lets us send the message in $H_2[\mathbf{p}]$. Can we get away with using less bits? Well, that would require some possible messages to share the same index. This would mean that, for those messages, the receiver wouldn't be able to recover the exact message we meant to send. Therefore, we cannot reduce the number of bits (without introducing some loss of meaning). In this sense, the entropy is a fundamental limit representing the "amount of information" in a source. At most, we can encode a message up to its entropy.

## Conclusion

I hope that this post shed some light regarding entropy, and its appearance in many fields. The key takeaway from this is that entropy is a very natural consequence that comes out of reasoning about how "likely" a set of probability distributions are. Given how general its derivation is, it shouldn't be so surprising that entropy pops up all over the place. Given its interpretation as the "likelihood of a distribution", it also intuitively makes sense why we often maximize it in various inference problems.

## References

#First edit.
#Next you can update your site name, avatar and other options using the _config.yml file in the root of your repository (shown below).

#![_config.yml]({{ site.baseurl }}/images/config.png)

#The easiest way to make your first post is to edit this one. Go into /_posts/ and update the Hello World markdown file. For more instructions head over to the [Jekyll Now #repository](https://github.com/barryclark/jekyll-now) on GitHub.
