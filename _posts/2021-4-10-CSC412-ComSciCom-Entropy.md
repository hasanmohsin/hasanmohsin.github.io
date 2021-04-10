---
layout: post
title: Trying to Understand Entropy
---

This post is going to try to answer the question: "What exactly is entropy"? 
Mathematically, the question is answered easily enough. Suppose we have some probability distribution $p(x)$ over a state space with $m$ outcomes $\{x_1, x_2, ... x_m\}$. In this post we will stick with a discrete state space, since the intuition is more clear and there are some complications with continuous ones. The entropy $$H$$ is defined for such a distribution as:

$$\begin{equation}
H[p] = - \sum_i p(x_i) \log p(x_i)
\end{equation}$$

Ok, but what actually *is* the Entropy? How can we understand it? It is the cornerstone of Information theory, underlies the 2nd law of thermodynamics, and appears all over the place in Machine Learning, so the task seems worthwhile. 

Generally it is described as some sort of measure of uncertainty, information, or surprise associated with a random event. It is also usually described as characterizing the width of a probability distribution. While these interpretations do offer some 

#First edit.
#Next you can update your site name, avatar and other options using the _config.yml file in the root of your repository (shown below).

#![_config.yml]({{ site.baseurl }}/images/config.png)

#The easiest way to make your first post is to edit this one. Go into /_posts/ and update the Hello World markdown file. For more instructions head over to the [Jekyll Now #repository](https://github.com/barryclark/jekyll-now) on GitHub.
