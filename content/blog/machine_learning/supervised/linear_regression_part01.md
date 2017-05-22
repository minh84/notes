+++
date          = "2017-05-18"
title         = "Linear Regression part 1"
type          = "subblog"
showonlyimage = true
draft         = false
author        = "Minh VU"
image         = "img/ln_reg_p01_logo.png"
description   = "Linear regression is one of the most basic ML algorithm but it's still very popular. This blog gives an introduction to linear regression."
+++


<div class="cell border-box-sizing text_cell rendered">
 <div class="prompt input_prompt">
 </div>
 <div class="inner_cell">
  <div class="text_cell_render border-box-sizing rendered_html">
   <p>
    In this notebook we will look at linear regression problem. Recall linear regression it to find $\theta$ such that 
$$
h(x,\theta) = \theta_0 + \theta_1x_1+\ldots+\theta_Dx_D
$$
is a good predictor for training set $(x^{(i)}, y^{(i)})_{i=1}^m$ i.e we want to find $\theta$ that minimize
$$
J(\theta) = \frac{1}{2}\sum_{i=1}^m \left(h(x^{(i)}, \theta)- y^{(i)}\right)^2
$$
where $J(\theta)$ is also call the
    <strong>
     least-square
    </strong>
    loss function.
   </p>
   <h2 id="Probabilistic-interpretation:-Maximum-likelihood">
    Probabilistic interpretation: Maximum-likelihood
    <a class="anchor-link" href="#Probabilistic-interpretation:-Maximum-likelihood">
     ¶
    </a>
   </h2>
   <p>
    In this part, we try to understand why we use the linear representation with the
    <strong>
     least-square
    </strong>
    error. Let's asssume the target and inputs are related via the equation
$$
y^{(i)} = h(x^{(i)},\theta) + \epsilon^{(i)}
$$
where $\epsilon^{(i)}$ is an error term that captures either un-modeled features or random noise. We assume further that $\epsilon^{(i)}\ \mathrm{i.i.d}\ \sim \mathcal{N}(0,\sigma^2)$.
   </p>
   <p>
    We have 
$$
p(\epsilon^{(i)}) = \frac{1}{\sqrt{2\pi}\sigma}\exp\left(-\frac{\left(\epsilon^{(i)}\right)^2}{2\sigma^2}\right)
$$
This implies that
$$
p(y^{(i)}|x^{(i)}) =  \frac{1}{\sqrt{2\pi}\sigma}\exp\left(-\frac{\left(y^{(i)} - h(x^{(i)},\theta)\right)^2}{2\sigma^2}\right)
$$
The likelihood function $L(\theta)$ is defined as
$$
\begin{array}{rl}
L(\theta) &amp;= p(y^{(i)},i=1,\ldots,m|x^{(i)},i=1,\ldots,m;\theta)\\
          &amp;=\prod_{i=1}^m p(y^{(i)}|x^{(i)};\theta)
\end{array}
$$
The maximum-likelihood is to find $\theta$ that maximize $L(\theta)$. By taking $-\log(L(\theta))$ it is equivalent to
$$
\mathrm{arg}\min_{\theta} \sum_{i=1}^m -\log\left(p(y^{(i)}|x^{(i)};\theta)\right)
$$
which can easily simplified to
$$
\mathrm{arg}\min_{\theta}\left( \mathrm{cst} +\frac{1}{2\sigma^2}\sum_{i=1}^m \left(y^{(i)}- h(x^{(i)},\theta)\right)^2\right) = \mathrm{arg}\min_{\theta} J(\theta)
$$
So the
    <strong>
     least-square
    </strong>
    loss function is actually result of maximum-likelihood method.
   </p>
   <h2 id="Gradient-descent-algorithm">
    Gradient descent algorithm
    <a class="anchor-link" href="#Gradient-descent-algorithm">
     ¶
    </a>
   </h2>
   <p>
    Gradient descent algorithm is well described in
    <a href="https://en.wikipedia.org/wiki/Gradient_descent">
     wikipedia
    </a>
    . The algorithm is based on the observation, if
$$
\theta_n = \theta_{n-1} -\lambda \nabla_{\theta} J(\theta)
$$
for $\lambda$ small enough, then we have
$$
J(\theta_{n-1})\geq J(\theta_{n})
$$
For least-square loss function, we have
$$
\nabla_{\theta} J(\theta) = \sum_{i=1}^m \left(h(x^{(i)}, \theta)- y^{(i)}\right)x^{(i)}
$$
So the update-rule becomes
$$
\theta_n = \theta_{n-1} -\lambda \sum_{i=1}^m \left(h(x^{(i)}, \theta_{n-1})- y^{(i)}\right)x^{(i)}
$$
Above update rule looks at all training examples, so it is also called
    <strong>
     batch gradient descent
    </strong>
    . This makes the method very expensive when $m$ is huge. So in practice, we often used
    <strong>
     mini-batch stochastic gradient descent
    </strong>
    (see
    <a href="https://en.wikipedia.org/wiki/Stochastic_gradient_descent">
     here
    </a>
    for detail description). In pseudocode, the method can be presented as follows
   </p>
   <div style="margin-left: 35px; margin-top: 15px; width: 600px">
    <div style="width:auto ; margin-left: margin-bottom:1.25em;border:1px solid #8898BF; background:transparent;padding:0">
     <div style="height:8px;margin:0;border:0;border-bottom:1px solid #8898BF;background: #C8D8FF;font-size:1px">
     </div>
     <div style="padding:5px;font-size:small">
      <ul>
       <li>
        choose an initial guess for $\theta_0$, learning rate $\lambda$ and batch-size $s$
       </li>
       <li>
        at each epoch, we randomly shuffle the training sample
        <ul>
         <li>
          we loop through each mini-batch samples of size $s$ i.e $(x^{(i)}, y^{(i)})_{i=k\cdot s}^{(k+1)\cdot s-1}$
         </li>
         <li>
          we compute the update rule for current mini-batch sample i.e
        $$
        \theta_n = \theta_{n-1} -\lambda \sum_{i=k\cdot s}^{(k+1)\cdot s-1} \left(h(x^{(i)}, \theta_{n-1})- y^{(i)}\right)x^{(i)}
        $$
         </li>
        </ul>
       </li>
      </ul>
     </div>
    </div>
   </div>
  </div>
  <h2 id="Linear-regression-examples">
   Linear regression examples
   <a class="anchor-link" href="#Linear-regression-examples">
    ¶
   </a>
  </h2>
  <p>
   It's time to implement linear regression with gradient descent using some synthetic data. We need to import the modules we need
  </p>
 </div>
</div>
<div class="cell border-box-sizing code_cell rendered">
 <div class="input">
  <div class="prompt input_prompt">
   In [1]:
  </div>
  <div class="inner_cell">
   <div class="input_area">
    <div class=" highlight hl-ipython3">
     <pre><span></span><span class="kn">import</span> <span class="nn">numpy</span> <span class="k">as</span> <span class="nn">np</span>
<span class="kn">import</span> <span class="nn">time</span><span class="o">,</span> <span class="nn">sys</span>

<span class="c1"># add parent to search path</span>
<span class="k">if</span> <span class="s1">'..'</span> <span class="ow">not</span> <span class="ow">in</span> <span class="n">sys</span><span class="o">.</span><span class="n">path</span><span class="p">:</span>
    <span class="n">sys</span><span class="o">.</span><span class="n">path</span><span class="o">.</span><span class="n">insert</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span> <span class="s1">'..'</span><span class="p">)</span>

    
<span class="c1"># for auto-reloading external modules</span>
<span class="o">%</span><span class="k">load_ext</span> autoreload
<span class="o">%</span><span class="k">autoreload</span> 2

<span class="c1"># imported helpers function   </span>
<span class="kn">from</span> <span class="nn">helpers</span> <span class="k">import</span> <span class="n">vis</span><span class="p">,</span> <span class="n">glm</span>
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
    We define a helper function to generate synthetic data
   </p>
  </div>
 </div>
</div>
<div class="cell border-box-sizing code_cell rendered">
 <div class="input">
  <div class="prompt input_prompt">
   In [2]:
  </div>
  <div class="inner_cell">
   <div class="input_area">
    <div class=" highlight hl-ipython3">
     <pre><span></span><span class="c1"># helper function to generate data</span>
<span class="k">def</span> <span class="nf">create_dataset</span><span class="p">(</span><span class="n">theta</span><span class="p">,</span> <span class="n">min_x</span><span class="p">,</span> <span class="n">max_x</span><span class="p">,</span> <span class="n">sigma</span><span class="p">,</span> <span class="n">N</span><span class="p">):</span>
    <span class="sd">'''</span>
<span class="sd">    We generate N sample randomly uniform between min_x, max_x</span>
<span class="sd">    then we append 1 to each sample</span>
<span class="sd">                  y[i] = x[i] * theta' + eps[i]</span>
<span class="sd">    where eps[i] ~ N(0, sigma^2)</span>
<span class="sd">    </span>
<span class="sd">    Input arguments</span>
<span class="sd">    :param theta: a ndarray of shape [D+1]</span>
<span class="sd">    :param min_x: min value for x of shape [D]</span>
<span class="sd">    :param max_x: max value for x of shape [D]</span>
<span class="sd">    :param sigma: standard deviation for error</span>
<span class="sd">    :param N: number of sample</span>
<span class="sd">    </span>
<span class="sd">    :return:</span>
<span class="sd">        X: a matrix of shape [N, D+1] (we append 1 at the beginning)</span>
<span class="sd">        y: a ndarray of shape [N]</span>
<span class="sd">    '''</span>
    <span class="n">D</span> <span class="o">=</span> <span class="nb">len</span><span class="p">(</span><span class="n">theta</span><span class="p">)</span> <span class="o">-</span> <span class="mi">1</span>
    <span class="n">sample_x</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">random</span><span class="o">.</span><span class="n">uniform</span><span class="p">(</span><span class="n">low</span><span class="o">=</span><span class="n">min_x</span><span class="p">,</span> <span class="n">high</span><span class="o">=</span><span class="n">max_x</span><span class="p">,</span> <span class="n">size</span><span class="o">=</span><span class="p">[</span><span class="n">N</span><span class="p">,</span> <span class="n">D</span><span class="p">])</span>
    <span class="n">X</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">hstack</span><span class="p">([</span><span class="n">np</span><span class="o">.</span><span class="n">ones</span><span class="p">([</span><span class="n">N</span><span class="p">,</span><span class="mi">1</span><span class="p">]),</span> <span class="n">sample_x</span><span class="p">])</span>
    <span class="n">y</span> <span class="o">=</span> <span class="n">X</span><span class="o">.</span><span class="n">dot</span><span class="p">(</span><span class="n">theta</span><span class="p">)</span> <span class="o">+</span> <span class="n">sigma</span> <span class="o">*</span> <span class="n">np</span><span class="o">.</span><span class="n">random</span><span class="o">.</span><span class="n">randn</span><span class="p">(</span><span class="n">N</span><span class="p">)</span>
    <span class="k">return</span> <span class="n">X</span><span class="p">,</span> <span class="n">y</span>
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
    We can generate training dataset and visualize it with
    <code>
     matplotlib
    </code>
   </p>
  </div>
 </div>
</div>
<div class="cell border-box-sizing code_cell rendered">
 <div class="input">
  <div class="prompt input_prompt">
   In [3]:
  </div>
  <div class="inner_cell">
   <div class="input_area">
    <div class=" highlight hl-ipython3">
     <pre><span></span><span class="kn">import</span> <span class="nn">matplotlib.pyplot</span> <span class="k">as</span> <span class="nn">plt</span>
<span class="o">%</span><span class="k">matplotlib</span> inline

<span class="n">in_theta</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">([</span><span class="mf">2.0</span><span class="p">,</span> <span class="mf">3.5</span><span class="p">])</span>
<span class="n">sigma</span> <span class="o">=</span> <span class="o">.</span><span class="mi">2</span>
<span class="n">min_x</span> <span class="o">=</span> <span class="p">[</span><span class="mf">0.</span><span class="p">]</span>
<span class="n">max_x</span> <span class="o">=</span> <span class="p">[</span><span class="mf">1.</span><span class="p">]</span>
<span class="n">N</span> <span class="o">=</span> <span class="mi">100</span>
<span class="n">data_X</span><span class="p">,</span> <span class="n">data_y</span> <span class="o">=</span> <span class="n">create_dataset</span><span class="p">(</span><span class="n">in_theta</span><span class="p">,</span> <span class="n">min_x</span><span class="p">,</span> <span class="n">max_x</span><span class="p">,</span> <span class="n">sigma</span><span class="p">,</span> <span class="n">N</span><span class="p">)</span>


<span class="n">vis</span><span class="o">.</span><span class="n">draw</span><span class="p">(</span><span class="n">data_X</span><span class="p">[:,</span><span class="mi">1</span><span class="p">],</span> <span class="n">data_y</span><span class="p">,</span> <span class="n">title</span><span class="o">=</span><span class="s1">'Training data generate with input theta=</span><span class="si">{}</span><span class="s1">'</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">in_theta</span><span class="p">))</span>

<span class="c1"># split to train &amp; validation</span>
<span class="kn">from</span> <span class="nn">sklearn.model_selection</span> <span class="k">import</span> <span class="n">train_test_split</span>
<span class="n">train_X</span><span class="p">,</span> <span class="n">val_X</span><span class="p">,</span> <span class="n">train_y</span><span class="p">,</span> <span class="n">val_y</span> <span class="o">=</span> <span class="n">train_test_split</span><span class="p">(</span><span class="n">data_X</span><span class="p">,</span> <span class="n">data_y</span><span class="p">,</span> <span class="n">test_size</span><span class="o">=</span><span class="mf">0.2</span><span class="p">)</span>
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
SUVORK5CYII=
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
    We have implemented GD/SGD fitting in
    <code>
     glm.py
    </code>
    , now let's try to train our model on generated synthetic dataset
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
     <pre><span></span><span class="n">clf</span> <span class="o">=</span> <span class="n">glm</span><span class="o">.</span><span class="n">Lsm</span><span class="p">()</span>

<span class="c1"># hyper parameters</span>
<span class="n">epochs</span> <span class="o">=</span> <span class="mi">100</span>

<span class="n">batch_size</span> <span class="o">=</span> <span class="mi">10</span>
<span class="n">clf</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">train_X</span><span class="p">,</span> <span class="n">train_y</span><span class="p">,</span> <span class="n">val_X</span><span class="p">,</span> <span class="n">val_y</span><span class="p">,</span> 
        <span class="n">epochs</span> <span class="o">=</span> <span class="n">epochs</span><span class="p">,</span> <span class="n">batch_size</span><span class="o">=</span><span class="n">batch_size</span><span class="p">,</span>
        <span class="n">solver</span> <span class="o">=</span> <span class="s1">'Sgd'</span><span class="p">,</span> <span class="n">learning_rate</span><span class="o">=</span><span class="mf">0.05</span><span class="p">,</span> <span class="n">debug</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span> 
        <span class="n">use_intercept</span> <span class="o">=</span> <span class="kc">False</span><span class="p">)</span>

<span class="c1"># plot loss function</span>
<span class="n">train_val_loss</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">(</span><span class="n">clf</span><span class="o">.</span><span class="n">_dbg_loss</span><span class="p">)</span>
<span class="n">nb_loss</span> <span class="o">=</span> <span class="n">train_val_loss</span><span class="o">.</span><span class="n">shape</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span>
<span class="n">_</span><span class="o">=</span> <span class="n">vis</span><span class="o">.</span><span class="n">draw</span><span class="p">(</span><span class="nb">range</span><span class="p">(</span><span class="n">nb_loss</span><span class="p">),</span> 
         <span class="n">train_val_loss</span><span class="p">,</span> 
         <span class="n">title</span> <span class="o">=</span> <span class="s1">'Fitted theta = </span><span class="si">{}</span><span class="s1">'</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">clf</span><span class="o">.</span><span class="n">_thetas</span><span class="p">),</span>
         <span class="n">legends</span> <span class="o">=</span> <span class="p">[</span><span class="s1">'trainning loss'</span><span class="p">,</span> <span class="s1">'validation loss'</span><span class="p">],</span> <span class="n">plot_type</span> <span class="o">=</span> <span class="s1">'plot'</span><span class="p">)</span>
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
<div class="cell border-box-sizing code_cell rendered">
 <div class="input">
  <div class="prompt input_prompt">
   In [18]:
  </div>
  <div class="inner_cell">
   <div class="input_area">
    <div class=" highlight hl-ipython3">
     <pre><span></span><span class="n">x_plot</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">linspace</span><span class="p">(</span><span class="n">min_x</span><span class="p">,</span> <span class="n">max_x</span><span class="p">,</span> <span class="mi">20</span><span class="p">)</span>
<span class="n">fig_ax</span> <span class="o">=</span> <span class="n">vis</span><span class="o">.</span><span class="n">draw</span><span class="p">(</span><span class="n">data_X</span><span class="p">[:,</span><span class="mi">1</span><span class="p">],</span> <span class="n">data_y</span><span class="p">,</span> <span class="n">plot_type</span><span class="o">=</span><span class="s1">'scatter'</span><span class="p">)</span>
<span class="n">_</span> <span class="o">=</span> <span class="n">vis</span><span class="o">.</span><span class="n">draw</span><span class="p">(</span><span class="n">x_plot</span><span class="p">,</span> <span class="n">x_plot</span> <span class="o">*</span> <span class="n">clf</span><span class="o">.</span><span class="n">_thetas</span><span class="p">[</span><span class="mi">1</span><span class="p">]</span> <span class="o">+</span> <span class="n">clf</span><span class="o">.</span><span class="n">_thetas</span><span class="p">[</span><span class="mi">0</span><span class="p">],</span> <span class="n">fig_ax</span> <span class="o">=</span> <span class="n">fig_ax</span><span class="p">,</span> 
         <span class="n">title</span> <span class="o">=</span> <span class="s1">'Training data generate with input theta=</span><span class="si">{}</span><span class="s1">'</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">in_theta</span><span class="p">),</span>
         <span class="n">plot_type</span><span class="o">=</span><span class="s1">'plot'</span><span class="p">,</span> <span class="n">colors</span><span class="o">=</span><span class="s1">'r'</span><span class="p">)</span>
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
BWHD2o9IAAAAAElFTkSuQmCC
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
    We can play around with various set of hyper-parameters
    <code>
     (batch_size, epochs, learning_rate)
    </code>
    , after trying various parameters one can notice the following
   </p>
   <ul>
    <li>
     the smaller batch_size the more noisy training loss, you can try batch_size = 5 for example
    </li>
    <li>
     the learning_rate for this dataset should be in [0.005, 0.1]
     <ul>
      <li>
       if the learning_rate is small e.g 1e-3, the training need ~ 500 epochs to converge
      </li>
      <li>
       if the learning_rate is big e.g 2e-1, the training will blow up
      </li>
     </ul>
    </li>
    <li>
     the fitting works much better with
     <code>
      SgdMomentum
     </code>
    </li>
   </ul>
   <p>
    For this simple dataset, we find that batch_size=10, epochs=100 and learning_rate=1e-2 seems performing well.
   </p>
   <h2 id="Conclusion">
    Conclusion
    <a class="anchor-link" href="#Conclusion">
     ¶
    </a>
   </h2>
   <p>
    The linear regression is one of the most simple learning algorithm, but it's still very popular since it's very fast to train and its result is easy to interpret. In this notebook, we have learnt to fit linear regression using Stochastic Gradient Descent (with/without momentum), however in practice, we often use the closed-form for linear regression. We will discuss it in next part.
   </p>
  </div>
 </div>
</div>
