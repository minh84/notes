+++
date          = "2017-05-18"
title         = "Machine Learning"
type          = "blog"
showonlyimage = true
draft         = false
author        = "Minh VU"
image         = "img/lego01-respondr.jpg"
description   = "Wonder what is Machine Learning and why it matters. I wonder it myself so I write this blog so I can share and learn more about Machine Learning."
+++

<div class="cell border-box-sizing text_cell rendered">
 <div class="prompt input_prompt">
 </div>
 <div class="inner_cell">
  <div class="text_cell_render border-box-sizing rendered_html">
   <h2 id="TOC">
    TOC
    <a class="anchor-link" href="#TOC">
     ¶
    </a>
   </h2>
   <ul>
    <li>
     <a href="#Introduction">
      Introduction
     </a>
    </li>
    <li>
     <a href="#Supervised-learning">
      Supervised learning
     </a>
     <ul>
      <li>
       <a href="#Linear-regression">
        Linear regression
       </a>
      </li>
     </ul>
    </li>
    <li>
     <a href="#Unsupervised-learning">
      Unsupervised learning
     </a>
     <ul>
      <li>
       <a href="#Restricted-Boltzmann-Machine">
        Restricted Boltzmann Machine
       </a>
      </li>
     </ul>
    </li>
   </ul>
   <h2 id="Introduction">
    Introduction
    <a class="anchor-link" href="#Introduction">
     ¶
    </a>
   </h2>
   <p>
    Machine learning (from
    <a href="https://en.wikipedia.org/wiki/Machine_learning">
     wikipedia
    </a>
    ) is the subfield of computer science that, according to Arthur Samuel in 1959, gives "computers the ability to learn without being explicitly programmed." Here learning means recognizing and understanding the input data then can make predictions on data.
   </p>
   <p>
    In this series of blogs, I would like to share with you a practical introduction to machine learning and statistical pattern recognition. This covers the three categories of machine learning
   </p>
   <p>
    <img alt="ml-categories" src="./assets/machine_learning-categories.jpg"/>
   </p>
   <h2 id="Supervised-learning">
    Supervised learning
    <a class="anchor-link" href="#Supervised-learning">
     ¶
    </a>
   </h2>
   <p>
    Supervised learning (from
    <a href="https://en.wikipedia.org/wiki/Supervised_learning">
     wikipedia
    </a>
    ) is the machine learning task of inferring a function from labeled training data. For example
   </p>
   <ul>
    <li>
     <p>
      given a training data contains the living areas and prices of some houses, we want to learn to predict the price of other houses in function of their living area.
     </p>
    </li>
    <li>
     <p>
      given a training data contains emails and their labels (spam/non-spam), we want to learn to predict whether new incoming email is spam/non-spam
     </p>
    </li>
   </ul>
   <p>
    Let's define some notation for future use
   </p>
   <ul>
    <li>
     $x^{(i)}\in\mathcal{X}$ denote the
     <code>
      input
     </code>
     variables, also called input
     <strong>
      features
     </strong>
    </li>
    <li>
     $y^{(i)}\in\mathcal{Y}$ denote the
     <code>
      output
     </code>
     or
     <strong>
      target
     </strong>
     variable that we want to predict
    </li>
    <li>
     a pair $(x^{(i)}, y^{(i)})$ is called a
     <strong>
      training example
     </strong>
    </li>
    <li>
     a list of $(x^{(i)}, y^{(i)}),i=1,\ldots,m$ is called a
     <strong>
      training set
     </strong>
    </li>
   </ul>
   <p>
    Our goal, given a training set, is to learn a function $h:\mathcal{X}\mapsto\mathcal{Y}$ so that $h(x)$ is a "good" predictor for the corresponding value of $y$. For historical reason, this function $h$ is call a
    <strong>
     hypothesis
    </strong>
    . The process is illustrated like this
   </p>
   <p>
    <img alt="supervised-learning-process" src="./assets/supervised-process.png" style="width: 300px;"/>
   </p>
   <p>
    The
    <strong>
     standard appoach
    </strong>
    to supervised learning problems is
   </p>
   <ul>
    <li>
     pick a representation for
     <strong>
      hypothesis
     </strong>
     function $h$
    </li>
    <li>
     pick a
     <strong>
      loss
     </strong>
     function $L(h(x), y)$ that we will minimize
    </li>
   </ul>
   <p>
    The supervised learning can divided into two categories
   </p>
   <ul>
    <li>
     when the target $y$ is continuous (e.g house price), we call it a
     <strong>
      regression
     </strong>
     problem
    </li>
    <li>
     when the target $y$ can only take discrete values (e.g spam/non-spam), we call it a
     <strong>
      classification
     </strong>
     problem
    </li>
   </ul>
   <h3 id="Linear-regression">
    Linear regression
    <a class="anchor-link" href="#Linear-regression">
     ¶
    </a>
   </h3>
   <p>
    Let's consider the case $\mathcal{X}=\mathbb{R}^D, \mathcal{Y}=\mathbb{R}$ and a linear representation of the input for our
    <strong>
     hypothesis
    </strong>
   </p>
   $$
h(x,\theta) = \theta_0 + \theta_1 x_1 +\ldots+\theta_Dx_D
$$
   <p>
    Then we pick least-square error as
    <strong>
     loss
    </strong>
    function
   </p>
   $$
J(\theta) = \frac{1}{2}\sum_{i=1}^m \left(h(x^{(i)},\theta) - y^{(i)}\right)^2
$$
   <p>
    The linear regression is discussed in the following notebooks
   </p>
   <ul>
    <li>
     Linear regression
     <a href="./supervised/linear_regression_part01/">
      part 1
     </a>
     which covers
     <ul>
      <li>
       probabilistic interpretation:
       <strong>
        maximum-likelihood (ML)
       </strong>
      </li>
      <li>
       iterative first order optimization algorithm: gradient descent
      </li>
     </ul>
    </li>
    <li>
     Linear regression
     <a href="./supervised/linear_regression_part02/">
      part 2
     </a>
     which covers
     <ul>
      <li>
       the normal equation: closed form for linear regression
      </li>
      <li>
       parameter regularization: Bayesian view and
       <strong>
        maximum-a-posteriori (MAP)
       </strong>
      </li>
     </ul>
    </li>
   </ul>
   <h2 id="Unsupervised-learning">
    Unsupervised learning
    <a class="anchor-link" href="#Unsupervised-learning">
     ¶
    </a>
   </h2>
   <p>
    Unsupervised learning (from
    <a href="https://en.wikipedia.org/wiki/Unsupervised_learning">
     wikipedia
    </a>
    is the machine learning task of inferring a function to describe hidden structure from "unlabeled" data.
   </p>
   <p>
    Approaches to unsupervised learning include:
   </p>
   <ul>
    <li>
     clustering
     <ul>
      <li>
       k-means
      </li>
     </ul>
    </li>
    <li>
     Neural Networks
     <ul>
      <li>
       Restricted Boltzmann Machine
      </li>
     </ul>
    </li>
   </ul>
   <h3 id="Restricted-Boltzmann-Machine">
    Restricted Boltzmann Machine
    <a class="anchor-link" href="#Restricted-Boltzmann-Machine">
     ¶
    </a>
   </h3>
   <p>
    Restricted Boltzman Machine (RBM) is energy-based model which consists of a layer of visible units and a layer of hidden units with no visible-visible or hidden-hidden connections.
   </p>
   <p>
    The implementation of RBMs is done in
    <a href="./unsupervised/rbm/">
     here
    </a>
    which covers
   </p>
   <ul>
    <li>
     Introduction to Restricted Boltzmann Machine (RBM)
    </li>
    <li>
     RBM feature learning to aplly in MNIST classification task
    </li>
   </ul>
  </div>
 </div>
</div>
