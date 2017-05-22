+++
date          = "2017-05-22"
title         = "Binary classfication with Logistic Regression"
type          = "subblog"
showonlyimage = true
draft         = false
author        = "Minh VU"
image         = "img/logistic_reg_logo.png"
description   = "We start by looking at binary classification using Logistic regression."
+++


<div class="cell border-box-sizing text_cell rendered">
 <div class="prompt input_prompt">
 </div>
 <div class="inner_cell">
  <div class="text_cell_render border-box-sizing rendered_html">
   <p>
    Let's now talk about classification and as an introduction we will focus on binary classifcation where $y$ can take on only two values, 0 and 1. For instance, if we are trying to build a spam classifier for email, then $x^{(i)}$ may be some features of an email and $y^{(i)}$ may be 1 if it is a spam mail, and 0 otherwise.
   </p>
   <p>
    In the context of classification, we often use the following notation
   </p>
   <ul>
    <li>
     $y^{(i)}$ is also called the
     <strong>
      label
     </strong>
     values
    </li>
    <li>
     0 is also called the
     <strong>
      negative
     </strong>
     class
    </li>
    <li>
     1 is also called the
     <strong>
      positive
     </strong>
     class
    </li>
   </ul>
   <p>
    In this notebook, we will look at logistic regression for binary classification task. First we import needed modules
   </p>
  </div>
 </div>
</div>
<div class="cell border-box-sizing code_cell rendered">
 <div class="input">
  <div class="prompt input_prompt">
   In [12]:
  </div>
  <div class="inner_cell">
   <div class="input_area">
    <div class=" highlight hl-ipython3">
     <pre><span></span><span class="c1"># import modules</span>
<span class="kn">import</span> <span class="nn">numpy</span> <span class="k">as</span> <span class="nn">np</span>
<span class="kn">import</span> <span class="nn">time</span><span class="o">,</span> <span class="nn">sys</span>

<span class="c1"># add parent to search path</span>
<span class="k">if</span> <span class="s1">'..'</span> <span class="ow">not</span> <span class="ow">in</span> <span class="n">sys</span><span class="o">.</span><span class="n">path</span><span class="p">:</span>
    <span class="n">sys</span><span class="o">.</span><span class="n">path</span><span class="o">.</span><span class="n">insert</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span> <span class="s1">'..'</span><span class="p">)</span>

    
<span class="c1"># for auto-reloading external modules</span>
<span class="o">%</span><span class="k">load_ext</span> autoreload
<span class="o">%</span><span class="k">autoreload</span> 2

<span class="c1"># imported helpers function   </span>
<span class="kn">from</span> <span class="nn">helpers</span> <span class="k">import</span> <span class="n">funcs</span><span class="p">,</span> <span class="n">vis</span><span class="p">,</span> <span class="n">glm</span>

<span class="c1"># matplotlib inline</span>
<span class="o">%</span><span class="k">matplotlib</span> inline
</pre>
    </div>
   </div>
  </div>
 </div>
 <div class="output_wrapper">
  <div class="output">
   <div class="output_area">
    <div class="prompt">
    </div>
    <div class="output_subarea output_stream output_stdout output_text">
     <pre>The autoreload extension is already loaded. To reload it, use:
  %reload_ext autoreload
</pre>
    </div>
   </div>
  </div>
 </div>
</div>
<div class="cell border-box-sizing text_cell rendered">
 <div class="prompt input_prompt">
 </div>
 <div class="inner_cell">
  <div class="text_cell_render border-box-sizing rendered_html">
   <h2 id="Logistic-regression">
    Logistic regression
    <a class="anchor-link" href="#Logistic-regression">
     ¶
    </a>
   </h2>
   <p>
    One could approach the classification above as a linear regression while ignoring the fact $y$ can only be either 0 or 1. However, this approach would perform poorly due to the following reasons
   </p>
   <ul>
    <li>
     the hypothesis function $h_\theta(x)$ is penalized if it is either much bigger than 1 or much smaller than 0
    </li>
    <li>
     $h_\theta(x)$ ignored the fact that $y$ can only be 0 or 1
    </li>
   </ul>
   <p>
    To fix that, we will choose
   </p>
   $$
h_\theta(x) = \frac{1}{1 + e^{-\theta^Tx}}
$$
   <p>
    where
   </p>
   $$
g(z) = \frac{1}{1+e^{-z}}
$$
   <p>
   </p>
   <p>
    is called the
    <strong>
     logistic function
    </strong>
    or the
    <strong>
     sigmoid function
    </strong>
    . Here is a plot showing $g(z)$
   </p>
  </div>
 </div>
</div>
<div class="cell border-box-sizing code_cell rendered">
 <div class="input">
  <div class="prompt input_prompt">
   In [13]:
  </div>
  <div class="inner_cell">
   <div class="input_area">
    <div class=" highlight hl-ipython3">
     <pre><span></span><span class="n">x</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">linspace</span><span class="p">(</span><span class="o">-</span><span class="mi">10</span><span class="p">,</span><span class="mi">10</span><span class="p">,</span><span class="mi">50</span><span class="p">)</span>
<span class="n">_</span> <span class="o">=</span> <span class="n">vis</span><span class="o">.</span><span class="n">draw</span><span class="p">(</span><span class="n">x</span><span class="p">,</span> <span class="n">funcs</span><span class="o">.</span><span class="n">sigmoid</span><span class="p">(</span><span class="n">x</span><span class="p">),</span> 
             <span class="n">plot_type</span><span class="o">=</span><span class="s1">'plot'</span><span class="p">,</span> <span class="n">title</span><span class="o">=</span><span class="s1">'sigmoid g(z)'</span><span class="p">,</span>
             <span class="n">xy_labels</span> <span class="o">=</span> <span class="p">[</span><span class="s1">'z'</span><span class="p">,</span> <span class="s1">'g(z)'</span><span class="p">])</span>
</pre>
    </div>
   </div>
  </div>
 </div>
 <div class="output_wrapper">
  <div class="output">
   <div class="output_area">
    <div class="prompt">
    </div>
    <div class="output_png output_subarea ">
     <img src="data:image/png;base64,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
pM3/B1y0ie08u8jKAAAAAElFTkSuQmCC
"/>
    </div>
   </div>
  </div>
 </div>
</div>
<div class="cell border-box-sizing text_cell rendered">
 <div class="prompt input_prompt">
 </div>
 <div class="inner_cell">
  <div class="text_cell_render border-box-sizing rendered_html">
   <p>
    The sigmoid function $g(z)$ has the following nice properties
   </p>
   <ul>
    <li>
     $g(z)\in [0, 1]$ for all $z\in\mathbb{R}$
    </li>
    <li>
     $g(z)$ is an increasing function and
    </li>
   </ul>
   \begin{split}
\lim_{z\rightarrow -\infty} g(z) &amp;= 0 \\
\lim_{z\rightarrow +\infty} g(z) &amp;= 1
\end{split}
   <ul>
    <li>
     $g(z)$ is continuous and its derivative is given by
    </li>
   </ul>
   $$
g'(z) = g(z)\left(1-g(z)\right)
$$
   <p>
    Now, we have chosen the hypothesis function, the next step is to define a loss function $J(\theta)$ given training samples $\left\{x^{(i)},\ y^{(i)}\right\}_{i=1}^m$.
   </p>
   <p>
    Let us assume that
   </p>
   \begin{split}
P\left(y=1|\ x;\theta\right) &amp;= h_\theta(x)\\
P\left(y=0|\ x;\theta\right) &amp;= 1 - h_\theta(x)
\end{split}
   <p>
    This can be written in a more compact form as
   </p>
   $$
P(y|x;\theta) = \left(h_\theta(x)\right)^y\left(1-h_\theta(x)\right)^{1-y}
$$
   <p>
    Assuming that all training samples are independent, then we write down the log-likelihood of the parameters as
   </p>
   \begin{split}
L(\theta) &amp;= \prod_{i=1}^{m}P\left(y^{(i)}|x^{(i)};\theta\right)\\
          &amp;= \prod_{i=1}^{m} \left(h_\theta(x^{(i)})\right)^{y^{(i)}} \left(1-h_\theta(x^{(i)})\right)^{1-y^{(i)}}
\end{split}
   <p>
    Then, we can use the method of maximize-log-likelihood
   </p>
   \begin{split}
\theta &amp;= \mathrm{arg}\max_{\theta}\log \left(L(\theta)\right) \\
&amp;= \mathrm{arg}\max_{\theta} \sum_{i=1}^{m}\left(y^{(i)}\log\left(h_\theta(x^{(i)} \right) + \left(1-y^{(i)}\right)\log\left(1-h_\theta(x^{(i)} \right)\right)
\end{split}
   <p>
    By taking negative log, we can define the loss function as
   </p>
   $$
J(\theta) = - \sum_{i=1}^{m}\left(y^{(i)}\log\left(h_\theta(x^{(i)} \right) + \left(1-y^{(i)}\right)\log\left(1-h_\theta(x^{(i)} \right)\right)
$$
   <p>
    And we want to find $\theta$ that minimizes $J(\theta)$, this can be solved by using
    <strong>
     SGD
    </strong>
   </p>
   $$
\theta:=\theta - \lambda \nabla_\theta J(\theta)
$$
   <p>
    To compute the gradient $\nabla_\theta J(\theta)$, we can just compute the derivative for the case of one training sample (then sum it up for general case)
   </p>
   \begin{split}
&amp;\nabla_\theta\left( y\log\left(h_\theta(x)\right) + (1-y)\log\left(1-h_\theta(x)\right) \right)\\
=&amp; \left(\frac{y}{h_\theta(x)} - \frac{1-y}{1 - h_\theta(x)}\right)\nabla_\theta h_\theta(x)\\
=&amp; \frac{y - h_\theta(x)}{h_\theta(x)\left(1-h_\theta(x)\right)} h_\theta(x)\left(1-h_\theta(x)\right) x\\
=&amp; \left(y - h_\theta(x)\right)x
\end{split}
   <p>
    This therefore gives us the update rule for a mini-batch $S$
   </p>
   $$
\theta_n = \theta_{n-1} - \lambda \sum_{i\in S} \left(h(x^{(i)}, \theta_{n-1})- y^{(i)}\right)x^{(i)}
$$
   <p>
    If we compare it with the update-rule for linear regression, we see it looks identical. It's supprising but actually logistic regression and linear regression are a sub-class of Generalized Linear Model.
   </p>
   <p>
    Before looking at GLM, let's implement logistic regression. We start by creating a synthetic dataset
   </p>
  </div>
 </div>
</div>
<div class="cell border-box-sizing code_cell rendered">
 <div class="input">
  <div class="prompt input_prompt">
   In [14]:
  </div>
  <div class="inner_cell">
   <div class="input_area">
    <div class=" highlight hl-ipython3">
     <pre><span></span><span class="n">X</span><span class="p">,</span> <span class="n">y</span> <span class="o">=</span> <span class="n">glm</span><span class="o">.</span><span class="n">demo_bin_class</span><span class="p">()</span>

<span class="n">neg_idx</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">where</span><span class="p">(</span><span class="n">y</span><span class="o">==</span><span class="mi">0</span><span class="p">)[</span><span class="mi">0</span><span class="p">]</span>
<span class="n">fig_ax</span> <span class="o">=</span> <span class="n">vis</span><span class="o">.</span><span class="n">draw</span><span class="p">(</span><span class="n">X</span><span class="p">[</span><span class="n">neg_idx</span><span class="p">,</span> <span class="mi">0</span><span class="p">],</span> <span class="n">X</span><span class="p">[</span><span class="n">neg_idx</span><span class="p">,</span> <span class="mi">1</span><span class="p">],</span> 
                  <span class="n">plot_type</span><span class="o">=</span><span class="s1">'scatter'</span><span class="p">,</span> <span class="n">colors</span><span class="o">=</span><span class="s1">'b'</span><span class="p">)</span>

<span class="n">pos_idx</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">where</span><span class="p">(</span><span class="n">y</span><span class="o">==</span><span class="mi">1</span><span class="p">)[</span><span class="mi">0</span><span class="p">]</span>
<span class="n">_</span> <span class="o">=</span> <span class="n">vis</span><span class="o">.</span><span class="n">draw</span><span class="p">(</span><span class="n">X</span><span class="p">[</span><span class="n">pos_idx</span><span class="p">,</span> <span class="mi">0</span><span class="p">],</span> <span class="n">X</span><span class="p">[</span><span class="n">pos_idx</span><span class="p">,</span> <span class="mi">1</span><span class="p">],</span> <span class="n">fig_ax</span><span class="o">=</span><span class="n">fig_ax</span><span class="p">,</span>
                  <span class="n">plot_type</span><span class="o">=</span><span class="s1">'scatter'</span><span class="p">,</span> <span class="n">colors</span><span class="o">=</span><span class="s1">'r'</span><span class="p">)</span>
</pre>
    </div>
   </div>
  </div>
 </div>
 <div class="output_wrapper">
  <div class="output">
   <div class="output_area">
    <div class="prompt">
    </div>
    <div class="output_png output_subarea ">
     <img src="data:image/png;base64,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
"/>
    </div>
   </div>
  </div>
 </div>
</div>
<div class="cell border-box-sizing text_cell rendered">
 <div class="prompt input_prompt">
 </div>
 <div class="inner_cell">
  <div class="text_cell_render border-box-sizing rendered_html">
   <p>
    Let's devide the dataset into training set and validation set (we will use sklearn to do this)
   </p>
  </div>
 </div>
</div>
<div class="cell border-box-sizing code_cell rendered">
 <div class="input">
  <div class="prompt input_prompt">
   In [15]:
  </div>
  <div class="inner_cell">
   <div class="input_area">
    <div class=" highlight hl-ipython3">
     <pre><span></span><span class="kn">from</span> <span class="nn">sklearn.model_selection</span> <span class="k">import</span> <span class="n">train_test_split</span>

<span class="n">X_train</span><span class="p">,</span> <span class="n">X_test</span><span class="p">,</span> <span class="n">y_train</span><span class="p">,</span> <span class="n">y_test</span> <span class="o">=</span> <span class="n">train_test_split</span><span class="p">(</span><span class="n">X</span><span class="p">,</span> <span class="n">y</span><span class="p">,</span> 
                                                    <span class="n">test_size</span><span class="o">=</span><span class="mf">0.1</span><span class="p">,</span> 
                                                    <span class="n">random_state</span><span class="o">=</span><span class="mi">42</span><span class="p">)</span>
</pre>
    </div>
   </div>
  </div>
 </div>
</div>
<div class="cell border-box-sizing text_cell rendered">
 <div class="prompt input_prompt">
 </div>
 <div class="inner_cell">
  <div class="text_cell_render border-box-sizing rendered_html">
   <p>
    The logistic regression is implemented in
    <code>
     glm.py
    </code>
    , here we will use it to fit the above dataset
   </p>
  </div>
 </div>
</div>
<div class="cell border-box-sizing code_cell rendered">
 <div class="input">
  <div class="prompt input_prompt">
   In [16]:
  </div>
  <div class="inner_cell">
   <div class="input_area">
    <div class=" highlight hl-ipython3">
     <pre><span></span><span class="c1"># create logistic regression model</span>
<span class="n">clf</span> <span class="o">=</span> <span class="n">glm</span><span class="o">.</span><span class="n">LogitReg</span><span class="p">(</span><span class="n">reg</span> <span class="o">=</span> <span class="mf">0.0</span><span class="p">)</span>

<span class="c1"># define some hyperparameters</span>
<span class="n">epochs</span> <span class="o">=</span> <span class="mi">50</span>
<span class="n">batch_size</span> <span class="o">=</span> <span class="mi">15</span>
<span class="n">learning_rate</span> <span class="o">=</span> <span class="mf">5e-2</span>

<span class="c1"># fit model with the data</span>
<span class="n">clf</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">X_train</span><span class="p">,</span> <span class="n">y_train</span><span class="p">,</span> <span class="n">X_test</span><span class="p">,</span> <span class="n">y_test</span><span class="p">,</span> 
        <span class="n">epochs</span> <span class="o">=</span> <span class="n">epochs</span><span class="p">,</span> <span class="n">batch_size</span> <span class="o">=</span> <span class="n">batch_size</span><span class="p">,</span> 
        <span class="n">learning_rate</span> <span class="o">=</span>  <span class="n">learning_rate</span><span class="p">,</span> <span class="n">solver</span> <span class="o">=</span> <span class="s1">'SgdMomentum'</span><span class="p">,</span>
        <span class="n">debug</span> <span class="o">=</span> <span class="kc">True</span><span class="p">)</span>

<span class="c1"># visualize training-loss vs validation-loss</span>
<span class="n">_</span> <span class="o">=</span> <span class="n">vis</span><span class="o">.</span><span class="n">draw</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">arange</span><span class="p">(</span><span class="nb">len</span><span class="p">(</span><span class="n">clf</span><span class="o">.</span><span class="n">_dbg_loss</span><span class="p">)),</span> <span class="n">clf</span><span class="o">.</span><span class="n">_dbg_loss</span><span class="p">,</span> 
             <span class="n">title</span> <span class="o">=</span> <span class="s1">'Prediction error in function of iterations'</span><span class="p">,</span>
             <span class="n">xy_labels</span><span class="o">=</span><span class="p">[</span><span class="s1">'iterations'</span><span class="p">,</span> <span class="s1">'error-rate'</span><span class="p">],</span>          
             <span class="n">legends</span><span class="o">=</span><span class="p">[</span><span class="s1">'training'</span><span class="p">,</span> <span class="s1">'validation'</span><span class="p">],</span>
             <span class="n">colors</span> <span class="o">=</span> <span class="p">[</span><span class="s1">'b'</span><span class="p">,</span><span class="s1">'y'</span><span class="p">],</span> <span class="n">plot_type</span> <span class="o">=</span> <span class="s1">'plot'</span><span class="p">)</span>

<span class="nb">print</span> <span class="p">(</span><span class="s1">'fitted thetas = </span><span class="si">{}</span><span class="s1">'</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">clf</span><span class="o">.</span><span class="n">_thetas</span><span class="p">))</span>
</pre>
    </div>
   </div>
  </div>
 </div>
 <div class="output_wrapper">
  <div class="output">
   <div class="output_area">
    <div class="prompt">
    </div>
    <div class="output_subarea output_stream output_stdout output_text">
     <pre>fitted thetas = [-4.47715455  4.76055688  4.82749919]
</pre>
    </div>
   </div>
   <div class="output_area">
    <div class="prompt">
    </div>
    <div class="output_png output_subarea ">
     <img src="data:image/png;base64,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
RU5ErkJggg==
"/>
    </div>
   </div>
  </div>
 </div>
</div>
<div class="cell border-box-sizing text_cell rendered">
 <div class="prompt input_prompt">
 </div>
 <div class="inner_cell">
  <div class="text_cell_render border-box-sizing rendered_html">
   <h3 id="Regularized-logistic-regression">
    Regularized logistic regression
    <a class="anchor-link" href="#Regularized-logistic-regression">
     ¶
    </a>
   </h3>
   <p>
    When fitting logistic regression, we notice that if $\theta$ fitted well to our training example then we have
   </p>
   $$
\left\{
\begin{split}
\theta^Tx^{(i)} &gt; 0 &amp; \text{ if } y^{(i)} = 1\\
\theta^Tx^{(i)} &lt; 0 &amp; \text{ otherwise }
\end{split}
\right.
$$
   <p>
    This is still true for any $\gamma \theta$ with $\gamma &gt; 0$. So to make the training more stable, it makes sense to add a regularization on $\theta$.
   </p>
  </div>
 </div>
</div>
<div class="cell border-box-sizing code_cell rendered">
 <div class="input">
  <div class="prompt input_prompt">
   In [17]:
  </div>
  <div class="inner_cell">
   <div class="input_area">
    <div class=" highlight hl-ipython3">
     <pre><span></span><span class="c1"># create logistic regression model with regularization</span>
<span class="n">clf</span> <span class="o">=</span> <span class="n">glm</span><span class="o">.</span><span class="n">LogitReg</span><span class="p">(</span><span class="n">reg</span> <span class="o">=</span> <span class="mf">5e-3</span><span class="p">)</span>

<span class="c1"># define some hyperparameters</span>
<span class="n">epochs</span> <span class="o">=</span> <span class="mi">50</span>
<span class="n">batch_size</span> <span class="o">=</span> <span class="mi">15</span>
<span class="n">learning_rate</span> <span class="o">=</span> <span class="mf">5e-2</span>

<span class="c1"># fit model with the data</span>
<span class="n">clf</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">X_train</span><span class="p">,</span> <span class="n">y_train</span><span class="p">,</span> <span class="n">X_test</span><span class="p">,</span> <span class="n">y_test</span><span class="p">,</span> 
        <span class="n">epochs</span> <span class="o">=</span> <span class="n">epochs</span><span class="p">,</span> <span class="n">batch_size</span> <span class="o">=</span> <span class="n">batch_size</span><span class="p">,</span> 
        <span class="n">learning_rate</span> <span class="o">=</span>  <span class="n">learning_rate</span><span class="p">,</span> <span class="n">solver</span> <span class="o">=</span> <span class="s1">'SgdMomentum'</span><span class="p">,</span>
        <span class="n">debug</span> <span class="o">=</span> <span class="kc">True</span><span class="p">)</span>

<span class="c1"># visualize training-loss vs validation-loss</span>
<span class="n">_</span> <span class="o">=</span> <span class="n">vis</span><span class="o">.</span><span class="n">draw</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">arange</span><span class="p">(</span><span class="nb">len</span><span class="p">(</span><span class="n">clf</span><span class="o">.</span><span class="n">_dbg_loss</span><span class="p">)),</span> <span class="n">clf</span><span class="o">.</span><span class="n">_dbg_loss</span><span class="p">,</span> 
             <span class="n">title</span> <span class="o">=</span> <span class="s1">'Prediction error in function of iterations'</span><span class="p">,</span>
             <span class="n">xy_labels</span><span class="o">=</span><span class="p">[</span><span class="s1">'iterations'</span><span class="p">,</span> <span class="s1">'error-rate'</span><span class="p">],</span>          
             <span class="n">legends</span><span class="o">=</span><span class="p">[</span><span class="s1">'training'</span><span class="p">,</span> <span class="s1">'validation'</span><span class="p">],</span>
             <span class="n">colors</span> <span class="o">=</span> <span class="p">[</span><span class="s1">'b'</span><span class="p">,</span><span class="s1">'y'</span><span class="p">],</span> <span class="n">plot_type</span> <span class="o">=</span> <span class="s1">'plot'</span><span class="p">)</span>

<span class="nb">print</span> <span class="p">(</span><span class="s1">'fitted thetas = </span><span class="si">{}</span><span class="s1">'</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">clf</span><span class="o">.</span><span class="n">_thetas</span><span class="p">))</span>
</pre>
    </div>
   </div>
  </div>
 </div>
 <div class="output_wrapper">
  <div class="output">
   <div class="output_area">
    <div class="prompt">
    </div>
    <div class="output_subarea output_stream output_stdout output_text">
     <pre>fitted thetas = [-3.12376821  3.57332084  3.40550646]
</pre>
    </div>
   </div>
   <div class="output_area">
    <div class="prompt">
    </div>
    <div class="output_png output_subarea ">
     <img src="data:image/png;base64,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
EwXDMDwbbrhh1wyoPSEuCvlMxd0XmCgUgImCYQw9Zs2axXXXXde1f/HFF3PppZey3377dU1zfd99
93U774033mC77bYDYN26dRx77LFsvfXWHH744WlzH5155plMnTqVbbfdlosuughwk+wtXbqUT33q
U3zqU58CUlNxA1x11VVst912bLfddlwdTQiVbYruYmLjFArARMEwSssrr5xHU1Nx584ePnwnNt88
80x7xxxzDOeddx5nnXUWAHfddRdz587l3HPPZb311mP58uXsvvvuHHLIIRnXP/7FL35BfX09L774
Is899xy77LJL17HLLruM0aNH09HRwX777cdzzz3Hueeey1VXXcXDDz/M2LFj09J6+umn+fWvf82T
Tz6JqrLbbrvxyU9+ksbGxryn6O4N5ikUgImCYQw9dt55Z95//32WLl3Ks88+S2NjIxMmTODb3/42
O+ywA/vvvz9vv/0272VZnP3RRx/tKpx32GEHdthhh65jd911F7vssgs777wzCxcu5IUXXshqz2OP
Pcbhhx9OQ0MDw4cP54gjjuDvf/87kP8U3b3BPIUCMFEwjNKS7Y2+lBx11FHcfffdvPvuuxxzzDHc
fvvtLFu2jKeffprq6momTZqUOGV2Ll5//XWuvPJKnnrqKRobG5kxY0aP0vHkO0V3byhbT6GtDa67
Lr2gv/56SLrHt90Gy5fbcpyGMVQ55phjuPPOO7n77rs56qijWLlyJeuvvz7V1dU8/PDD/Oc//8l6
/j777NM1qd6CBQt47rnnAFi1ahUNDQ2MHDmS9957L21yvUxTdu+9997ce++9rF27ljVr1vCnP/2J
vffeu4i5zU7ZegpXXQWzZkFVFXzpS/Dcc3DWWTBhAkSLIwGwbBnMmOEExK+9YaJgGEOLbbfdltWr
V7PRRhuxwQYbcMIJJ/D5z3+e7bffnqlTp7LVVltlPf/MM8/klFNOYeutt2brrbdmypQpAOy4447s
vPPObLXVVmy88cbsueeeXefMnDmT6dOns+GGG/Lwww93he+yyy7MmDGDadOmAXDaaaex8847l6Sq
KImyFYV33nHb3jPwHl1cuFtbU8e952aiYBhDj+eff75re+zYsTzxxBOJ8ZqamgDXW8hPmT1s2DDu
vPPOxPjh9Nkh55xzDuecc07Xfljof/WrX+WrX/1qWvzwepA+RXcxKdvqIy8CdXXu2xf+0fPuwlcv
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
VBmJ/AAAAABJRU5ErkJggg==
"/>
    </div>
   </div>
  </div>
 </div>
</div>
<div class="cell border-box-sizing text_cell rendered">
 <div class="prompt input_prompt">
 </div>
 <div class="inner_cell">
  <div class="text_cell_render border-box-sizing rendered_html">
   <p>
    Using the regularized form, we notice that the parameters are smaller and more stable (we obtain the similar parameter after each run).
   </p>
   <h3 id="Decision-boundary">
    Decision boundary
    <a class="anchor-link" href="#Decision-boundary">
     ¶
    </a>
   </h3>
   <p>
    Now, given the model is fitted, let's draw a decision boundary. First we consider the prediction defined as following equation
   </p>
   $$
\left\{\begin{split}
y = 1 &amp; \text{ if } h_{\theta}(x) &gt; 0.5\\
y = 0 &amp; \text{ otherwise}.
\end{split}\right.
$$
   <p>
    Then the decision boundary is defined as
   </p>
   $$B = \left\{x:\theta^Tx=0\right\}$$
  </div>
 </div>
</div>
<div class="cell border-box-sizing code_cell rendered">
 <div class="input">
  <div class="prompt input_prompt">
   In [22]:
  </div>
  <div class="inner_cell">
   <div class="input_area">
    <div class=" highlight hl-ipython3">
     <pre><span></span><span class="n">thetas</span> <span class="o">=</span> <span class="n">clf</span><span class="o">.</span><span class="n">_thetas</span>
<span class="n">x_b</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">linspace</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">20</span><span class="p">)</span>
<span class="n">y_b</span> <span class="o">=</span> <span class="p">(</span><span class="o">-</span><span class="n">thetas</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span> <span class="o">-</span> <span class="n">thetas</span><span class="p">[</span><span class="mi">1</span><span class="p">]</span><span class="o">*</span><span class="n">x_b</span><span class="p">)</span><span class="o">/</span><span class="n">thetas</span><span class="p">[</span><span class="mi">2</span><span class="p">]</span>

<span class="n">neg_idx</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">where</span><span class="p">(</span><span class="n">y</span><span class="o">==</span><span class="mi">0</span><span class="p">)[</span><span class="mi">0</span><span class="p">]</span>
<span class="n">fig_ax</span> <span class="o">=</span> <span class="n">vis</span><span class="o">.</span><span class="n">draw</span><span class="p">(</span><span class="n">X</span><span class="p">[</span><span class="n">neg_idx</span><span class="p">,</span> <span class="mi">0</span><span class="p">],</span> <span class="n">X</span><span class="p">[</span><span class="n">neg_idx</span><span class="p">,</span> <span class="mi">1</span><span class="p">],</span> 
                  <span class="n">plot_type</span><span class="o">=</span><span class="s1">'scatter'</span><span class="p">,</span> <span class="n">colors</span><span class="o">=</span><span class="s1">'b'</span><span class="p">)</span>

<span class="n">pos_idx</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">where</span><span class="p">(</span><span class="n">y</span><span class="o">==</span><span class="mi">1</span><span class="p">)[</span><span class="mi">0</span><span class="p">]</span>
<span class="n">_</span> <span class="o">=</span> <span class="n">vis</span><span class="o">.</span><span class="n">draw</span><span class="p">(</span><span class="n">X</span><span class="p">[</span><span class="n">pos_idx</span><span class="p">,</span> <span class="mi">0</span><span class="p">],</span> <span class="n">X</span><span class="p">[</span><span class="n">pos_idx</span><span class="p">,</span> <span class="mi">1</span><span class="p">],</span> <span class="n">fig_ax</span><span class="o">=</span><span class="n">fig_ax</span><span class="p">,</span>
             <span class="n">plot_type</span><span class="o">=</span><span class="s1">'scatter'</span><span class="p">,</span> <span class="n">colors</span><span class="o">=</span><span class="s1">'r'</span><span class="p">)</span>

<span class="n">_</span> <span class="o">=</span> <span class="n">vis</span><span class="o">.</span><span class="n">draw</span><span class="p">(</span><span class="n">x_b</span><span class="p">,</span> <span class="n">y_b</span><span class="p">,</span> <span class="n">fig_ax</span><span class="o">=</span><span class="n">fig_ax</span><span class="p">,</span>
             <span class="n">plot_type</span><span class="o">=</span><span class="s1">'plot'</span><span class="p">,</span> <span class="n">colors</span><span class="o">=</span><span class="s1">'g'</span><span class="p">)</span>
</pre>
    </div>
   </div>
  </div>
 </div>
 <div class="output_wrapper">
  <div class="output">
   <div class="output_area">
    <div class="prompt">
    </div>
    <div class="output_png output_subarea ">
     <img src="data:image/png;base64,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
"/>
    </div>
   </div>
  </div>
 </div>
</div>
<div class="cell border-box-sizing text_cell rendered">
 <div class="prompt input_prompt">
 </div>
 <div class="inner_cell">
  <div class="text_cell_render border-box-sizing rendered_html">
   <p>
    We see that boundary separates the two classes but it's closer to the blue ones than the red ones. We will revisite the optimal boundary when we study
    <strong>
     SVM
    </strong>
    .
   </p>
   <h2 id="Generalized-Linear-Models">
    Generalized Linear Models
    <a class="anchor-link" href="#Generalized-Linear-Models">
     ¶
    </a>
   </h2>
   <p>
    In this section, we introduce the GLM family which can be applied to a range of linear regression and logistic regression. We begin by defining  exponential family distributions $\text{ExpFam}(\eta)$ are the set of distributions in the following form
   </p>
   $$
 \left\{p(y;\eta) = b(y)\exp\left(\eta^TT(y) - a(\eta)\right)\right\}
$$
   <p>
    where
   </p>
   <ul>
    <li>
     $\eta$ is called the
     <strong>
      natural parameters
     </strong>
     of the distribution
    </li>
    <li>
     $T(y)$ is called the
     <strong>
      sufficient statistics
     </strong>
     , it's often that $T(y)=y$
    </li>
    <li>
     $a(\eta)$ is the
     <strong>
      log partition fundtion
     </strong>
     , the quantity $e^{-a(\eta)}$ essentially plays the role of a normalization constant
    </li>
   </ul>
   <h3 id="Example-of-exponential-distributions">
    Example of exponential distributions
    <a class="anchor-link" href="#Example-of-exponential-distributions">
     ¶
    </a>
   </h3>
   <h4 id="Bernoulli-distribution">
    Bernoulli distribution
    <a class="anchor-link" href="#Bernoulli-distribution">
     ¶
    </a>
   </h4>
   <p>
    We have the Bernoulli distribution
   </p>
   $$
\begin{split}
p(y,\phi) &amp;= \phi^y(1-\phi)^{1-y}\\
&amp;= \exp\left(\log\left(\frac{\phi}{1-\phi}\right)y + \log(1-\phi)\right)
\end{split}
$$
   <p>
    Thus, the Bernoulli distribution is in exponential distribution family with
   </p>
   $$
\left\{\begin{split}
T(y) &amp;= y\\
\eta &amp;= \log\left(\frac{\phi}{1-\phi}\right)\\
a(\eta) &amp;= -\log(1-\phi) = \log(1+e^{\eta})\\
b(y)&amp;=1
\end{split}\right.
$$
   <h4 id="Gaussian-distribution">
    Gaussian distribution
    <a class="anchor-link" href="#Gaussian-distribution">
     ¶
    </a>
   </h4>
   <p>
    The Guassian distribution is defined by
   </p>
   $$
\begin{split}
p(y;\mu,\sigma^2) &amp;= \frac{1}{\sqrt{2\pi}\sigma}\exp\left(-\frac{(y-\mu)^2}{2\sigma^2}\right)\\
&amp;= \frac{1}{\sqrt{2\pi}\sigma}\exp\left(-\frac{y^2}{2\sigma^2}\right)\exp\left(\frac{\mu y}{\sigma^2} - \frac{\mu^2}{2\sigma^2}\right)
\end{split}
$$
   <p>
    Thus, the Gaussian is is in exponential distribution family with
$$
\left\{\begin{split}
T(y) &amp;= \frac{y}{\sigma}\\
\eta &amp;= \frac{\mu}{\sigma}\\
a(\eta) &amp;= \frac{\mu^2}{2\sigma^2} = \frac{\eta^2}{2}\\
b(y)&amp;=\frac{1}{\sqrt{2\pi}\sigma}\exp\left(-\frac{y^2}{2\sigma^2}\right)
\end{split}\right.
$$
   </p>
   <h3 id="Constructing-GLMs">
    Constructing GLMs
    <a class="anchor-link" href="#Constructing-GLMs">
     ¶
    </a>
   </h3>
   <p>
    The GLMs makes three following assumptions
   </p>
   <ol>
    <li>
     <p>
      $y|x;\theta\sim \text{ExpFam}(\eta)$ i.e given $x$ and $\theta$ the distribution of $y$ follows some exponential family distribution
     </p>
    </li>
    <li>
     <p>
      Given $x$, our
      <strong>
       goal
      </strong>
      is to predict the expected value of $T(y)$ given $x$ i.e we define the
      <strong>
       hypothesis
      </strong>
      function $h_\theta(x)$ as
$$
h_\theta(x) = \mathbb{E}\left[T(y)|x;\theta\right]
$$
     </p>
    </li>
    <li>
     <p>
      The natural parameter $\eta$ is given by
$$
\eta = \theta^T x
$$
     </p>
    </li>
   </ol>
   <h4 id="Least-squares-derived-by-GLMs">
    Least squares derived by GLMs
    <a class="anchor-link" href="#Least-squares-derived-by-GLMs">
     ¶
    </a>
   </h4>
   <p>
    Consider $y|x;\theta \sim \mathcal{N}\left(\mu,\sigma^2\right)$, we have
   </p>
   $$
\begin{split}
h_\theta(x) &amp;= \mathbb{E}\left[T(y)|x;\theta\right]\\
&amp;= \mathbb{E}\left[\frac{y}{\sigma}|x;\theta\right]\\
&amp;= \frac{\mu}{\sigma} = \eta\\
&amp;= \theta^Tx\\
\end{split}
$$
   <h4 id="Logistic-regression-derived-by-GLMs">
    Logistic regression derived by GLMs
    <a class="anchor-link" href="#Logistic-regression-derived-by-GLMs">
     ¶
    </a>
   </h4>
   <p>
    Consider $y|x;\theta \sim \mathrm{Bernoulli}(\phi)$, we have
$$
\begin{split}
h_\theta(x) &amp;= \mathbb{E}\left[T(y)|x;\theta\right]\\
            &amp;= \mathbb{E}\left[y|x;\theta\right]\\
            &amp;= p(y=1|x;\theta)\\
            &amp;= \phi = \frac{1}{1+exp(-\eta)}\\
            &amp;= \frac{1}{1+exp(-\theta^Tx)}
\end{split}
$$
   </p>
   <h3 id="Fitting-GLMs">
    Fitting GLMs
    <a class="anchor-link" href="#Fitting-GLMs">
     ¶
    </a>
   </h3>
   <p>
    To fit GLMs, we can use maximum-log-likelihood
   </p>
   $$
\begin{split}
\theta &amp;= \mathrm{arg}\max_{\theta} \sum_{i=1}^m \log p\left(y^{(i)}\left|x^{(i)};\theta\right.\right)\\
&amp;= \mathrm{arg}\max_{\theta} \sum_{i=1}^m \left(\log b(y^{(i)})    + \theta^Tx^{(i)}T(y^{(i)}) - a\left(\theta^Tx^{(i)}\right)\right)
\end{split}
$$
   <p>
    In the case that $b(y)$ doesn't depend on $\eta$ (e.g Bernoulli and Gaussian), we can simplify above equation to
   </p>
   $$
\mathrm{arg}\max_{\theta} \sum_{i=1}^m \left(\theta^Tx^{(i)}T(y^{(i)}) - a\left(\theta^Tx^{(i)}\right)\right)
$$
   <p>
    Note that the derivation with respect to $\theta$ for one element of the above sum is given by
   </p>
   $$
\begin{split}
\nabla_\theta\left(\theta^TxT(y) - a\left(\theta^Tx\right)\right) &amp;= T(y)x - a'\left(\theta^Tx\right)x\\
&amp;= \left(T(y) - a'\left(\theta^Tx\right)\right) x
\end{split}
$$
   <p>
    For both Least-square and Logistic-regression, we have
$$
T(y) - a'\left(\theta^Tx\right) = cst \times \left(y-h_{\theta}(x)\right)
$$
   </p>
   <p>
    This is the reason that both Least-square and Logistic-regression has the same
    <strong>
     GD
    </strong>
    update-rule.
   </p>
   <h2 id="Conclusion">
    Conclusion
    <a class="anchor-link" href="#Conclusion">
     ¶
    </a>
   </h2>
   <p>
    We have learnt how the Logistic regression works in theory and practice. We notice that
   </p>
   <ul>
    <li>
     Logistic regression should be fitted using SgdMomentum otherwise it doesn't fit well
    </li>
    <li>
     Logistic regression should be regularized
    </li>
   </ul>
   <p>
    We also look at GLMs as general framework for both linear regression and logistic regression. Note that GLMs can be applied for other learning algorithms such as Softmax for e.g.
   </p>
  </div>
 </div>
</div>
