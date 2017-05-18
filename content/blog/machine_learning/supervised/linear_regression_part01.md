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
   <bdi>In</bdi>&nbsp;[3]:
  </div>
  <div class="inner_cell">
   <div class="input_area">
    <div class=" highlight hl-ipython3">
     <pre><span></span><span class="kn">import</span> <span class="nn">numpy</span> <span class="k">as</span> <span class="nn">np</span>
<span class="kn">import</span> <span class="nn">time</span>
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
   <bdi>In</bdi>&nbsp;[4]:
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
   In [5]:
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
<span class="n">_</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">scatter</span><span class="p">(</span><span class="n">data_X</span><span class="p">[:,</span><span class="mi">1</span><span class="p">],</span> <span class="n">data_y</span><span class="p">)</span>
<span class="n">_</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span><span class="s1">'Training data generate with input theta=</span><span class="si">{}</span><span class="s1">'</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">in_theta</span><span class="p">))</span>

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
Q8jB3cxsCP1/sT8En3jRQeUAAAAASUVORK5CYII=
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
    Let's implement the GD/SGD fitting to recover original $\theta$ from training data
   </p>
  </div>
 </div>
</div>
<div class="cell border-box-sizing code_cell rendered">
 <div class="input">
  <div class="prompt input_prompt">
   In [6]:
  </div>
  <div class="inner_cell">
   <div class="input_area">
    <div class=" highlight hl-ipython3">
     <pre><span></span><span class="k">class</span> <span class="nc">LinearRegressionModel</span><span class="p">(</span><span class="nb">object</span><span class="p">):</span>
    <span class="k">def</span> <span class="nf">__init__</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">learning_rate</span><span class="p">):</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">_learning_rate</span> <span class="o">=</span> <span class="n">learning_rate</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">_theta</span> <span class="o">=</span> <span class="kc">None</span>
    
    <span class="k">def</span> <span class="nf">_loss</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">batch_x</span><span class="p">,</span> <span class="n">batch_y</span><span class="p">):</span>
        <span class="sd">'''</span>
<span class="sd">        compute mini-batch loss and derivative with respect to theta</span>
<span class="sd">        </span>
<span class="sd">        Input</span>
<span class="sd">        :param batch_x is ndarray of shape [batch_size, D+1]</span>
<span class="sd">        :param batch_y is ndarray of shape [batch_size]</span>
<span class="sd">        </span>
<span class="sd">        :return:</span>
<span class="sd">            loss: least-square loss function</span>
<span class="sd">            dloss/dtheta: derivative of loss with respect to theta</span>
<span class="sd">        '''</span>
        <span class="k">assert</span> <span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">_theta</span> <span class="ow">is</span> <span class="ow">not</span> <span class="kc">None</span><span class="p">)</span>
                
        <span class="n">err</span> <span class="o">=</span> <span class="n">batch_x</span><span class="o">.</span><span class="n">dot</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">_theta</span><span class="p">)</span> <span class="o">-</span> <span class="n">batch_y</span>
        <span class="n">loss</span> <span class="o">=</span> <span class="mf">0.5</span> <span class="o">*</span> <span class="n">np</span><span class="o">.</span><span class="n">sum</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">square</span><span class="p">(</span><span class="n">err</span><span class="p">))</span>
        <span class="n">dtheta</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">transpose</span><span class="p">(</span><span class="n">batch_x</span><span class="p">)</span><span class="o">.</span><span class="n">dot</span><span class="p">(</span><span class="n">err</span><span class="p">)</span>
        
        <span class="k">return</span> <span class="n">loss</span><span class="p">,</span> <span class="n">dtheta</span>
    
    <span class="k">def</span> <span class="nf">_step</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">batch_X</span><span class="p">,</span> <span class="n">batch_y</span><span class="p">):</span>
        <span class="n">loss</span><span class="p">,</span> <span class="n">dtheta</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">_loss</span><span class="p">(</span><span class="n">batch_X</span><span class="p">,</span> <span class="n">batch_y</span><span class="p">)</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">_theta</span>  <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">_theta</span> <span class="o">-</span> <span class="bp">self</span><span class="o">.</span><span class="n">_learning_rate</span> <span class="o">*</span> <span class="n">dtheta</span>
        <span class="k">return</span> <span class="n">loss</span>
    
    <span class="k">def</span> <span class="nf">get_batches</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">train_X</span><span class="p">,</span> <span class="n">train_y</span><span class="p">,</span> <span class="n">batch_size</span><span class="p">):</span>
        <span class="c1"># get number of sample and compute number of batches</span>
        <span class="n">N</span> <span class="o">=</span> <span class="n">train_X</span><span class="o">.</span><span class="n">shape</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span>
        <span class="n">nb_batches</span>   <span class="o">=</span> <span class="n">N</span> <span class="o">//</span> <span class="n">batch_size</span>
        
        <span class="c1"># shuffle training sample</span>
        <span class="n">idx</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">arange</span><span class="p">(</span><span class="n">N</span><span class="p">)</span>
        <span class="n">np</span><span class="o">.</span><span class="n">random</span><span class="o">.</span><span class="n">shuffle</span><span class="p">(</span><span class="n">idx</span><span class="p">)</span>
        
        <span class="c1"># return batch_x, batch_y</span>
        <span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">nb_batches</span><span class="p">):</span>            
            <span class="n">batch_idx</span> <span class="o">=</span> <span class="n">idx</span><span class="p">[</span><span class="n">i</span><span class="o">*</span><span class="n">batch_size</span><span class="p">:(</span><span class="n">i</span><span class="o">+</span><span class="mi">1</span><span class="p">)</span> <span class="o">*</span> <span class="n">batch_size</span><span class="p">]</span>
            <span class="k">yield</span> <span class="n">train_X</span><span class="p">[</span><span class="n">batch_idx</span><span class="p">],</span> <span class="n">train_y</span><span class="p">[</span><span class="n">batch_idx</span><span class="p">]</span>
            
    
    <span class="k">def</span> <span class="nf">fit</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">train_X</span><span class="p">,</span> <span class="n">train_y</span><span class="p">,</span> <span class="n">val_X</span><span class="p">,</span> <span class="n">val_y</span><span class="p">,</span>
            <span class="n">batch_size</span><span class="p">,</span> <span class="n">epochs</span><span class="p">,</span> <span class="n">debug</span> <span class="o">=</span> <span class="kc">False</span><span class="p">):</span>
        <span class="c1"># get number of sample and input dimension</span>
        <span class="n">N</span><span class="p">,</span> <span class="n">input_dim</span> <span class="o">=</span> <span class="n">train_X</span><span class="o">.</span><span class="n">shape</span>
        <span class="n">nb_batches</span>   <span class="o">=</span> <span class="n">N</span> <span class="o">//</span> <span class="n">batch_size</span>
        <span class="c1"># initialized by a random-normal</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">_theta</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">random</span><span class="o">.</span><span class="n">randn</span><span class="p">(</span><span class="n">input_dim</span><span class="p">)</span>
        
        <span class="bp">self</span><span class="o">.</span><span class="n">_dbg_loss</span> <span class="o">=</span> <span class="kc">None</span>
        <span class="c1"># if debug on, we store loss        </span>
        <span class="k">if</span> <span class="n">debug</span><span class="p">:</span>
            <span class="bp">self</span><span class="o">.</span><span class="n">_dbg_loss</span> <span class="o">=</span> <span class="p">[]</span>
        <span class="c1"># iteratively update theta        </span>
        <span class="k">for</span> <span class="n">e</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">epochs</span><span class="p">):</span>            
            <span class="k">for</span> <span class="n">batch_X</span><span class="p">,</span> <span class="n">batch_y</span> <span class="ow">in</span> <span class="bp">self</span><span class="o">.</span><span class="n">get_batches</span><span class="p">(</span><span class="n">train_X</span><span class="p">,</span> <span class="n">train_y</span><span class="p">,</span> <span class="n">batch_size</span><span class="p">):</span>
                <span class="n">loss</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">_step</span><span class="p">(</span><span class="n">batch_X</span><span class="p">,</span> <span class="n">batch_y</span><span class="p">)</span>
                <span class="k">if</span> <span class="n">debug</span><span class="p">:</span>    
                    <span class="n">val_loss</span><span class="p">,</span><span class="n">_</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">_loss</span><span class="p">(</span><span class="n">val_X</span><span class="p">,</span> <span class="n">val_y</span><span class="p">)</span> 
                    <span class="bp">self</span><span class="o">.</span><span class="n">_dbg_loss</span><span class="o">.</span><span class="n">append</span><span class="p">([</span><span class="n">loss</span><span class="o">/</span><span class="n">batch_size</span><span class="p">,</span> <span class="n">val_loss</span><span class="o">/</span><span class="n">val_X</span><span class="o">.</span><span class="n">shape</span><span class="p">[</span><span class="mi">0</span><span class="p">]])</span>
    
    <span class="k">def</span> <span class="nf">predict</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">x</span><span class="p">):</span>
        <span class="k">assert</span> <span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">_theta</span> <span class="ow">is</span> <span class="ow">not</span> <span class="kc">None</span><span class="p">)</span>
        <span class="k">return</span> <span class="n">x</span><span class="o">.</span><span class="n">dot</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">_theta</span><span class="p">)</span>
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
    We can try to train our model on generated synthetic dataset
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
     <pre><span></span><span class="n">clf</span> <span class="o">=</span> <span class="n">LinearRegressionModel</span><span class="p">(</span><span class="n">learning_rate</span><span class="o">=</span><span class="mf">0.01</span><span class="p">)</span>

<span class="c1"># hyper parameters</span>
<span class="n">epochs</span> <span class="o">=</span> <span class="mi">100</span>

<span class="n">batch_size</span> <span class="o">=</span> <span class="mi">10</span>
<span class="n">clf</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">train_X</span><span class="p">,</span> <span class="n">train_y</span><span class="p">,</span> <span class="n">val_X</span><span class="p">,</span> <span class="n">val_y</span><span class="p">,</span> <span class="n">batch_size</span><span class="p">,</span> <span class="n">epochs</span><span class="p">,</span> <span class="n">debug</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>

<span class="c1"># plot loss function</span>
<span class="n">train_val_loss</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">(</span><span class="n">clf</span><span class="o">.</span><span class="n">_dbg_loss</span><span class="p">)</span>
<span class="n">nb_loss</span> <span class="o">=</span> <span class="n">train_val_loss</span><span class="o">.</span><span class="n">shape</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span>
<span class="n">plt</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="nb">range</span><span class="p">(</span><span class="n">nb_loss</span><span class="p">),</span> <span class="n">train_val_loss</span><span class="p">[:,</span><span class="mi">0</span><span class="p">],</span> <span class="n">label</span> <span class="o">=</span> <span class="s1">'trainning loss'</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="nb">range</span><span class="p">(</span><span class="n">nb_loss</span><span class="p">),</span> <span class="n">train_val_loss</span><span class="p">[:,</span><span class="mi">1</span><span class="p">],</span> <span class="n">label</span> <span class="o">=</span> <span class="s1">'validation loss'</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">legend</span><span class="p">()</span>
<span class="n">_</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span><span class="s1">'Fitted theta = </span><span class="si">{}</span><span class="s1">'</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">clf</span><span class="o">.</span><span class="n">_theta</span><span class="p">))</span>
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
   In [11]:
  </div>
  <div class="inner_cell">
   <div class="input_area">
    <div class=" highlight hl-ipython3">
     <pre><span></span><span class="n">_</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">scatter</span><span class="p">(</span><span class="n">data_X</span><span class="p">[:,</span><span class="mi">1</span><span class="p">],</span> <span class="n">data_y</span><span class="p">)</span>
<span class="n">x_plot</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">linspace</span><span class="p">(</span><span class="n">min_x</span><span class="p">,</span> <span class="n">max_x</span><span class="p">,</span> <span class="mi">20</span><span class="p">)</span>
<span class="n">_</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">x_plot</span><span class="p">,</span> <span class="n">x_plot</span> <span class="o">*</span> <span class="n">clf</span><span class="o">.</span><span class="n">_theta</span><span class="p">[</span><span class="mi">1</span><span class="p">]</span> <span class="o">+</span> <span class="n">clf</span><span class="o">.</span><span class="n">_theta</span><span class="p">[</span><span class="mi">0</span><span class="p">],</span> <span class="n">c</span><span class="o">=</span><span class="s1">'r'</span><span class="p">)</span>
<span class="n">_</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span><span class="s1">'Training data generate with input theta=</span><span class="si">{}</span><span class="s1">'</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">in_theta</span><span class="p">))</span>
              
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
AAAAAElFTkSuQmCC
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
    The linear regression is one of the most simple learning algorithm, but it's still very popular since it's very fast to train and its result is very easy to interpret. In this notebook, we have learnt to fit linear regression using Stochastic Gradient Descent, however in practice, we often use the closed-form for linear regression. We will discuss it in next part.
   </p>
  </div>
 </div>
</div>
