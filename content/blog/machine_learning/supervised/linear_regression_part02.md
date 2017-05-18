+++
date          = "2017-05-18"
title         = "Linear Regression part 2"
type          = "subblog"
showonlyimage = true
draft         = false
author        = "Minh VU"
image         = "img/ln_reg_p02_logo.png"
description   = "We continue to look at linear regression, this blog walks you though the Normal equation and coefficients' confident interval. We also look at regualized linear regression with maximum-a-posteriori."
+++


<div class="cell border-box-sizing text_cell rendered">
 <div class="prompt input_prompt">
 </div>
 <div class="inner_cell">
  <div class="text_cell_render border-box-sizing rendered_html">
   <p>
    In this notebook, we continue to look at linear regression problem
$$
\mathrm{arg}\min_{\theta}J(\theta) = \frac{1}{2}\sum_{i=1}^m \left(h(x^{(i)}, \theta)- y^{(i)}\right)^2
$$
where 
$$
h(x,\theta) = \theta_0 + \theta_1x_1+\ldots+\theta_Dx_D
$$
   </p>
   <h2 id="The-normal-equations">
    The normal equations
    <a class="anchor-link" href="#The-normal-equations">
     ¶
    </a>
   </h2>
   <p>
    We define the
    <strong>
     design matrix
    </strong>
    $\textbf{X}$ to be $m\times (D+1)$ matrix that contains training input values in its rows i.e
$$
\textbf{X} = \left(\begin{array}{cc}
1 &amp; \left(x^{(1)}\right)^T\\
\vdots\\
1 &amp; \left(x^{(m)}\right)^T\\
\end{array}\right)
$$
Also $\textbf{y}$ be the $m$-dimensional vector that contains training target values.
$$
\textbf{y} = \left(\begin{array}{c}
 y^{(1)}\\
\vdots\\
y^{(m)}\\
\end{array}\right)
$$
   </p>
   <p>
    Then the
    <strong>
     least-square
    </strong>
    loss function can be re-written as follows
$$
J(\theta) = \frac{1}{2}\left(\textbf{y} - \textbf{X}\theta\right)^T\left(\textbf{y} - \textbf{X}\theta\right)
$$
We have
$$
\begin{split}
\nabla_{\theta}J(\theta) &amp;= \frac{1}{2} \left(\theta^T\textbf{X}^T\textbf{X}\theta - 2 y^T\textbf{X}\theta + y^Ty\right)\\
&amp;= \frac{1}{2} \left(\textbf{X}^T\textbf{X}\theta + \textbf{X}^T\textbf{X}\theta - 2\textbf{X}^Ty\right)\\
&amp;= \textbf{X}^T\textbf{X}\theta - \textbf{X}^Ty
\end{split}
$$
where we use the two following derivations
   </p>
   \begin{split}
\nabla_{\theta}\left(\theta^TA\theta\right) &amp;= (A+A^T)\theta\\
\nabla_{\theta}\left(A\theta\right) &amp;= A^T
\end{split}
   <p>
    To minimize $J(\theta)$, we could set its derivatives to zero $\nabla_{\theta}J(\theta) = 0$ which implies
    <strong>
     the normal equations
    </strong>
    $$
\textbf{X}^T\textbf{X}\theta - \textbf{X}^Ty = 0
$$
Thus, the $\theta$ that minimizes $J(\theta)$ is given in closed form by the equation
$$
\theta = \left(\textbf{X}^T\textbf{X}\right)^{-1}\textbf{X}^Ty
$$
   </p>
   <h3 id="Parameters'-statistics">
    Parameters' statistics
    <a class="anchor-link" href="#Parameters'-statistics">
     ¶
    </a>
   </h3>
   <p>
    Recall the assumption that we have 
$$
y^{(i)} = h(x^{(i)},\theta^*) + \epsilon^{(i)}
$$
where $\theta^*$ is the real model's $\theta$. This can re-written in matrix form
$$
\textbf{y} = \textbf{X}\theta^* + \epsilon
$$
Plug this into the closed form of fitted $\theta$ (denoted by $\theta^f$)
$$
\theta^f = \left(\textbf{X}^T\textbf{X}\right)^{-1}\textbf{X}^T(\textbf{X}\theta^* + \epsilon) = \theta^* + \left(\textbf{X}^T\textbf{X}\right)^{-1} \textbf{X}^T\epsilon
$$
so the $\theta^f$ is an un-biased estimation of $\theta^*$ i.e
$$
\mathbb{E}\left[\theta^f\right] = \theta^*
$$
and the variance of $\theta^f$ is given by the equation
\begin{split}
\mathbb{V}\left[\theta^f\right] &amp;= \mathbb{E}\left[\left(\textbf{X}^T\textbf{X}\right)^{-1}\textbf{X}^T \epsilon \epsilon^T \textbf{X}\left(\textbf{X}^T\textbf{X}\right)^{-1}\right] \
&amp;= \sigma^2 \left(\textbf{X}^T\textbf{X}\right)^{-1}\textbf{X}^T \textbf{X}\left(\textbf{X}^T\textbf{X}\right)^{-1}\
&amp;= \sigma^2 \left(\textbf{X}^T\textbf{X}\right)^{-1}
\end{split}
since we have $\mathbb{E}\left[\epsilon \epsilon^T\right] = \sigma^2 \mathrm{Id}_{m}$ (where $\mathrm{Id}_{m}$ is the identity matrix of size $m\times m$). Thus we have
$$
\theta^f \sim \mathcal{N}\left(\theta^*, \sigma^2 \left(\textbf{X}^T\textbf{X}\right)^{-1}\right)
$$
Typically, one estimates the variance $\sigma^2$ by
$$
\hat{\sigma}^2= \frac{1}{m-D-1} \sum_{i=1}^m \left(h(x^{(i)},\theta^f) - y^{(i)}\right)^2
$$
This allows us to
   </p>
   <ul>
    <li>
     test if a particular coefficient $\theta_j=0$ using
     <em>
      Z-score
     </em>
     $$
z_j = \frac{\theta^f_j}{\hat{\sigma}\sqrt{v_j}}
$$
where $v_j$ is the $j-th$ diagonal element of $\left(\textbf{X}^T\textbf{X}\right)^{-1}$. Under the null hypothesis that $\theta_j=0$, $z_j$ is distributed as $t_{N-D-1}$ ($t$ student distribution with $N-p-1$ degrees of freedom), and hence a large (absolute) value of $z_j$ will leed to rejection of this null hypothesis.
    </li>
    <li>
     estimate $1-2\alpha$ confidence interval of coefficient:
$$
\theta^*_j \in (\theta^f_j - \mathcal{N}^{-1}(1-\alpha)\hat{\sigma}\sqrt{v_j}, \theta^f_j + \mathcal{N}^{-1}(1-\alpha)\hat{\sigma}\sqrt{v_j})
$$
where $\alpha=0.025$ gives 95% confidence interval:
$$
\theta^*_j \in (\theta^f_j - 1.96\hat{\sigma}\sqrt{v_j}, \theta^f_j + 1.96\hat{\sigma}\sqrt{v_j})
$$
    </li>
   </ul>
   <p>
    Let's implement linear regression using the normal equations. We start by loading needed modules
   </p>
  </div>
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
<span class="kn">import</span> <span class="nn">sys</span>

<span class="c1"># add parent to search path</span>
<span class="k">if</span> <span class="s1">'..'</span> <span class="ow">not</span> <span class="ow">in</span> <span class="n">sys</span><span class="o">.</span><span class="n">path</span><span class="p">:</span>
    <span class="n">sys</span><span class="o">.</span><span class="n">path</span><span class="o">.</span><span class="n">insert</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span> <span class="s1">'..'</span><span class="p">)</span>

    
<span class="c1"># for auto-reloading external modules</span>
<span class="o">%</span><span class="k">load_ext</span> autoreload
<span class="o">%</span><span class="k">autoreload</span> 2

<span class="c1"># imported helpers function   </span>
<span class="kn">from</span> <span class="nn">helpers</span> <span class="k">import</span> <span class="n">linear_regression</span>

<span class="c1"># create a synthetic dataset</span>
<span class="n">in_theta</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">([</span><span class="mf">2.0</span><span class="p">,</span> <span class="mf">3.5</span><span class="p">])</span>
<span class="n">sigma</span> <span class="o">=</span> <span class="o">.</span><span class="mi">2</span>
<span class="n">min_x</span> <span class="o">=</span> <span class="p">[</span><span class="mf">0.</span><span class="p">]</span>
<span class="n">max_x</span> <span class="o">=</span> <span class="p">[</span><span class="mf">1.</span><span class="p">]</span>
<span class="n">N</span> <span class="o">=</span> <span class="mi">100</span>
<span class="n">data_X</span><span class="p">,</span> <span class="n">data_y</span> <span class="o">=</span> <span class="n">linear_regression</span><span class="o">.</span><span class="n">create_dataset</span><span class="p">(</span><span class="n">in_theta</span><span class="p">,</span> <span class="n">min_x</span><span class="p">,</span> <span class="n">max_x</span><span class="p">,</span> <span class="n">sigma</span><span class="p">,</span> <span class="n">N</span><span class="p">)</span>
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
    The implementation is straightforward and you can look at
    <code>
     linear_regression.py
    </code>
    for more detail. We now try it with synthetic dataset
   </p>
  </div>
 </div>
</div>
<div class="cell border-box-sizing code_cell rendered">
 <div class="input">
  <div class="prompt input_prompt">
   In [8]:
  </div>
  <div class="inner_cell">
   <div class="input_area">
    <div class=" highlight hl-ipython3">
     <pre><span></span><span class="nb">print</span><span class="p">(</span><span class="s1">'original theta </span><span class="si">{}</span><span class="se">\n</span><span class="s1">'</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">in_theta</span><span class="p">))</span>
<span class="n">clf</span> <span class="o">=</span> <span class="n">linear_regression</span><span class="o">.</span><span class="n">LinearRegressionModel</span><span class="p">()</span>
<span class="n">clf</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">data_X</span><span class="p">,</span> <span class="n">data_y</span><span class="p">)</span>

<span class="n">clf</span><span class="o">.</span><span class="n">summary</span><span class="p">()</span>
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
     <pre>original theta [ 2.   3.5]

R2-score 0.965

Fitted parameters
coef       fitted    F-score    low 95%    high 95%
-------  --------  ---------  ---------  ----------
theta_0   1.95342    50.2202    1.87719     2.02966
theta_1   3.60002    52.5588    3.46577     3.73427
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
    Looking at above result we see
   </p>
   <ul>
    <li>
     F-score is high: so we can reject null hypothesis
    </li>
    <li>
     $\theta^f$ is closed to original $\theta^*$
    </li>
    <li>
     confidence interval contains the true $\theta^*$ as expected
    </li>
   </ul>
   <h3 id="Pros-and-cons">
    Pros and cons
    <a class="anchor-link" href="#Pros-and-cons">
     ¶
    </a>
   </h3>
   <p>
    Using the normal equations has quite a few advantage
   </p>
   <ul>
    <li>
     it's fast and accurate
    </li>
    <li>
     it returns useful statitics
    </li>
    <li>
     it can handle huge data: we can notice that the only thing we need is $X^TX$ and $X^Ty$, this can be computed iteratively for huge data as long as number of features input is not big.
    </li>
   </ul>
   <p>
    However, it also has some weekness
   </p>
   <ul>
    <li>
     solving the normal equations can be un-stable when data is skewed (big feature values and small features values), one can scale input features to same range before using the fit
    </li>
   </ul>
   <h2 id="Bayesian-view-and-maximum-a-posteriori-(MAP)">
    Bayesian view and maximum-a-posteriori (MAP)
    <a class="anchor-link" href="#Bayesian-view-and-maximum-a-posteriori-(MAP)">
     ¶
    </a>
   </h2>
   <p>
    From beginning until now, we approach the linear regression problem using maximum-likelihood, this approach assumes $\theta$ is a fixed parameters. An alternative way to approach our parameter estimation problems is to
take the
    <strong>
     Bayesian
    </strong>
    view of the world where $\theta$ is a random variable whose value is unknown. In this approach, we would specify a prior distribution $p(\theta)$ on $\theta$ that expresses our “prior beliefs” about the
parameters.
   </p>
   <p>
    Given a training set $S = \left\{(x^{(i)}, y^{(i)})\right\}_{i=1}^m$, we can compute the posteriori
$$
p(\theta|S) = \frac{p(S|\theta)p(\theta)}{p(S)}
$$
The
    <strong>
     maximum-a-posteriori (MAP)
    </strong>
    approach assumes
   </p>
   <ul>
    <li>
     $p(S)$ does not depend on any particular $\theta$
    </li>
    <li>
     $p(S|\theta) = \prod_{i=1}^m p((x^{(i)}, y^{(i)})|\theta)$
    </li>
   </ul>
   <p>
    Then
    <strong>
     MAP
    </strong>
    estimates $\theta$ such that
$$
\theta = \mathrm{arg}\max_{\theta} \left(\prod_{i=1}^m p\left((x^{(i)}, y^{(i)})|\theta\right) \right) p(\theta)
$$
   </p>
   <h3 id="MAP-as-Ridge-regression">
    MAP as Ridge regression
    <a class="anchor-link" href="#MAP-as-Ridge-regression">
     ¶
    </a>
   </h3>
   <p>
    For linear regression, let's consider a special case:
   </p>
   <ul>
    <li>
     prior distribution $\theta \sim \mathcal{N}(0, \tau^2\mathrm{Id}_{D+1})$
    </li>
    <li>
     and $y^{(i)}  = h(x^{(i)}, \theta) + \epsilon^{(i)}$ where $\epsilon^{(i)}$ are i.i.d $\sim \mathcal{N}(0,\sigma^2)$
    </li>
   </ul>
   <p>
    Then by taking $-\log$ on $\left(\prod_{i=1}^m p\left((x^{(i)}, y^{(i)})|\theta\right) \right) p(\theta)$ we obtain
$$\begin{split}
\theta &amp;= \mathrm{arg}\min_{\theta} \sum_{i=1}^m -\log p\left((x^{(i)}, y^{(i)})|\theta\right) - \log p(\theta)\\
&amp;= \mathrm{arg}\min_{\theta}\sum_{i=1}^m\left(cst + \frac{\left(y^{(i)}- h(x^{(i)},\theta)\right)^2}{2\sigma^2}\right) + \sum_{j=0}^{D}\left(cst + \frac{\theta_j^2}{2\tau^2}\right)\\
&amp;= \mathrm{arg}\min_{\theta} \sum_{i=1}^m \frac{\left(y^{(i)}- h(x^{(i)},\theta)\right)^2}{2\sigma^2} + \sum_{j=0}^{D}\frac{\theta_j^2}{2\tau^2}\\
&amp;= \mathrm{arg}\min_{\theta} \sum_{i=1}^m \left(y^{(i)}- h(x^{(i)},\theta)\right)^2 + \frac{\sigma^2}{\tau^2} \sum_{j=0}^{D}\theta_j^2
\end{split}
$$
Denote $r=\frac{\sigma^2}{\tau^2}$, the
    <strong>
     MAP
    </strong>
    estimators becomes
$$
\mathrm{arg}\min_{\theta} \sum_{i=1}^m \left(y^{(i)}- h(x^{(i)},\theta)\right)^2 + r\times \sum_{j=0}^{D}\theta_j^2
$$
The above form is also called
    <strong>
     Ridge
    </strong>
    regression which is a form of parameter regularisation. Intuitively if $\tau$ is small i.e "prior beliefs" believes that $\theta$ is closed to zeros, then we will have $r$ is big which pernalizes large value of $\theta$.
   </p>
   <p>
    The
    <strong>
     MAP
    </strong>
    estimators can be written in matrix form
$$
\mathrm{arg}\min_{\theta} \left(\textbf{y}-\textbf{X}\theta \right)^T\left(\textbf{y} -\textbf{X}\theta\right) + r\times \theta^T\theta
$$
By taking derivatives as previous step, we also obtain closed form given by
$$
\theta^{\mathrm{MAP}} = \left(\textbf{X}^T\textbf{X} + r\times \mathrm{Id}_{D+1}\right)^{-1}\textbf{X}^Ty
$$
   </p>
   <p>
    Note that, the regularization term often ignores the intercept $\theta_0$, this can easily done by first substract both $x^{(i)}, y^{(i)}$ by its empirical mean $\bar{x}, \bar{y}$ and do the optimization without the intercept to obtain $\theta_1,...,\theta_D$ then the intercept is given by equation
$$
\theta_0 = \bar{y} - \bar{x}^T\left(\begin{array}{c}\theta_1\\ \vdots\\ \theta_D\end{array}\right)
$$
   </p>
   <p>
    First, we create and visualize a synthetic dataset
   </p>
  </div>
 </div>
</div>
<div class="cell border-box-sizing code_cell rendered">
 <div class="input">
  <div class="prompt input_prompt">
   In [25]:
  </div>
  <div class="inner_cell">
   <div class="input_area">
    <div class=" highlight hl-ipython3">
     <pre><span></span><span class="n">theta</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">([</span><span class="mf">1.0</span><span class="p">,</span> <span class="o">-</span><span class="mf">1.0</span><span class="p">,</span> <span class="o">-</span><span class="mf">2.0</span><span class="p">,</span> <span class="mf">1.0</span><span class="p">])</span>
<span class="n">min_x</span> <span class="o">=</span> <span class="o">-</span><span class="mf">2.0</span>
<span class="n">max_x</span> <span class="o">=</span> <span class="mf">2.0</span>
<span class="n">sigma</span> <span class="o">=</span> <span class="mf">0.3</span>
<span class="n">N</span> <span class="o">=</span> <span class="mi">50</span>
<span class="n">poly_X</span><span class="p">,</span> <span class="n">poly_y</span> <span class="o">=</span> <span class="n">linear_regression</span><span class="o">.</span><span class="n">create_poly_dataset</span><span class="p">(</span><span class="n">theta</span><span class="p">,</span> <span class="n">min_x</span><span class="p">,</span> <span class="n">max_x</span><span class="p">,</span> <span class="n">sigma</span><span class="p">,</span> <span class="n">N</span><span class="p">,</span> <span class="n">deg</span> <span class="o">=</span> <span class="mi">6</span><span class="p">)</span>

<span class="kn">import</span> <span class="nn">matplotlib.pyplot</span> <span class="k">as</span> <span class="nn">plt</span>
<span class="o">%</span><span class="k">matplotlib</span> inline

<span class="n">_</span><span class="o">=</span><span class="n">plt</span><span class="o">.</span><span class="n">scatter</span><span class="p">(</span><span class="n">poly_X</span><span class="p">[:,</span><span class="mi">1</span><span class="p">],</span> <span class="n">poly_y</span><span class="p">)</span>
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
    Now let's fit our polynomial using Ridge regression with $r=0$ (i.e no regularization) with various degree
   </p>
  </div>
 </div>
</div>
<div class="cell border-box-sizing code_cell rendered">
 <div class="input">
  <div class="prompt input_prompt">
   In [49]:
  </div>
  <div class="inner_cell">
   <div class="input_area">
    <div class=" highlight hl-ipython3">
     <pre><span></span><span class="n">clf</span> <span class="o">=</span> <span class="n">linear_regression</span><span class="o">.</span><span class="n">RidgeRegressionModel</span><span class="p">(</span><span class="n">reg</span> <span class="o">=</span> <span class="mf">0.</span><span class="p">)</span>
<span class="n">x</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">linspace</span><span class="p">(</span><span class="n">min_x</span><span class="p">,</span> <span class="n">max_x</span><span class="p">,</span> <span class="mi">20</span><span class="p">)</span>

<span class="n">plt</span><span class="o">.</span><span class="n">figure</span><span class="p">(</span><span class="mi">1</span><span class="p">)</span>
<span class="n">deg</span> <span class="o">=</span> <span class="mi">1</span>
<span class="n">clf</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">poly_X</span><span class="p">[:,:</span><span class="n">deg</span><span class="o">+</span><span class="mi">1</span><span class="p">],</span> <span class="n">poly_y</span><span class="p">)</span>
<span class="n">_</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">scatter</span><span class="p">(</span><span class="n">poly_X</span><span class="p">[:,</span><span class="mi">1</span><span class="p">],</span> <span class="n">poly_y</span><span class="p">)</span>
<span class="n">_</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">x</span><span class="p">,</span> <span class="n">clf</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">x</span><span class="p">))</span>
<span class="n">_</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span><span class="s1">'Fit using polynomial degree=</span><span class="si">{}</span><span class="s1">'</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">deg</span><span class="p">))</span>

<span class="n">plt</span><span class="o">.</span><span class="n">figure</span><span class="p">(</span><span class="mi">2</span><span class="p">)</span>
<span class="n">deg</span> <span class="o">=</span> <span class="mi">2</span>
<span class="n">clf</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">poly_X</span><span class="p">[:,:</span><span class="n">deg</span><span class="o">+</span><span class="mi">1</span><span class="p">],</span> <span class="n">poly_y</span><span class="p">)</span>
<span class="n">_</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">scatter</span><span class="p">(</span><span class="n">poly_X</span><span class="p">[:,</span><span class="mi">1</span><span class="p">],</span> <span class="n">poly_y</span><span class="p">)</span>
<span class="n">_</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">x</span><span class="p">,</span> <span class="n">clf</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">x</span><span class="p">))</span>
<span class="n">_</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span><span class="s1">'Fit using polynomial degree=</span><span class="si">{}</span><span class="s1">'</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">deg</span><span class="p">))</span>

<span class="n">plt</span><span class="o">.</span><span class="n">figure</span><span class="p">(</span><span class="mi">3</span><span class="p">)</span>
<span class="n">deg</span> <span class="o">=</span> <span class="mi">3</span>
<span class="n">clf</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">poly_X</span><span class="p">[:,:</span><span class="n">deg</span><span class="o">+</span><span class="mi">1</span><span class="p">],</span> <span class="n">poly_y</span><span class="p">)</span>
<span class="n">_</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">scatter</span><span class="p">(</span><span class="n">poly_X</span><span class="p">[:,</span><span class="mi">1</span><span class="p">],</span> <span class="n">poly_y</span><span class="p">)</span>
<span class="n">_</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">x</span><span class="p">,</span> <span class="n">clf</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">x</span><span class="p">))</span>
<span class="n">_</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span><span class="s1">'Fit using polynomial degree=</span><span class="si">{}</span><span class="s1">'</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">deg</span><span class="p">))</span>

<span class="n">plt</span><span class="o">.</span><span class="n">figure</span><span class="p">(</span><span class="mi">4</span><span class="p">)</span>
<span class="n">deg</span> <span class="o">=</span> <span class="mi">4</span>
<span class="n">clf</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">poly_X</span><span class="p">[:,:</span><span class="n">deg</span><span class="o">+</span><span class="mi">1</span><span class="p">],</span> <span class="n">poly_y</span><span class="p">)</span>
<span class="n">_</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">scatter</span><span class="p">(</span><span class="n">poly_X</span><span class="p">[:,</span><span class="mi">1</span><span class="p">],</span> <span class="n">poly_y</span><span class="p">)</span>
<span class="n">_</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">x</span><span class="p">,</span> <span class="n">clf</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">x</span><span class="p">))</span>
<span class="n">_</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span><span class="s1">'Fit using polynomial degree=</span><span class="si">{}</span><span class="s1">'</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">deg</span><span class="p">))</span>

<span class="n">plt</span><span class="o">.</span><span class="n">figure</span><span class="p">(</span><span class="mi">5</span><span class="p">)</span>
<span class="n">deg</span> <span class="o">=</span> <span class="mi">5</span>
<span class="n">clf</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">poly_X</span><span class="p">[:,:</span><span class="n">deg</span><span class="o">+</span><span class="mi">1</span><span class="p">],</span> <span class="n">poly_y</span><span class="p">)</span>
<span class="n">_</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">scatter</span><span class="p">(</span><span class="n">poly_X</span><span class="p">[:,</span><span class="mi">1</span><span class="p">],</span> <span class="n">poly_y</span><span class="p">)</span>
<span class="n">_</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">x</span><span class="p">,</span> <span class="n">clf</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">x</span><span class="p">))</span>
<span class="n">_</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span><span class="s1">'Fit using polynomial degree=</span><span class="si">{}</span><span class="s1">'</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">deg</span><span class="p">))</span>

<span class="n">plt</span><span class="o">.</span><span class="n">figure</span><span class="p">(</span><span class="mi">6</span><span class="p">)</span>
<span class="n">deg</span> <span class="o">=</span> <span class="mi">6</span>
<span class="n">clf</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">poly_X</span><span class="p">[:,:</span><span class="n">deg</span><span class="o">+</span><span class="mi">1</span><span class="p">],</span> <span class="n">poly_y</span><span class="p">)</span>
<span class="n">_</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">scatter</span><span class="p">(</span><span class="n">poly_X</span><span class="p">[:,</span><span class="mi">1</span><span class="p">],</span> <span class="n">poly_y</span><span class="p">)</span>
<span class="n">_</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">x</span><span class="p">,</span> <span class="n">clf</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">x</span><span class="p">))</span>
<span class="n">_</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span><span class="s1">'Fit using polynomial degree=</span><span class="si">{}</span><span class="s1">'</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">deg</span><span class="p">))</span>
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
o64AAAAASUVORK5CYII=
"/>
    </div>
   </div>
   <div class="output_area">
    <div class="prompt">
    </div>
    <div class="output_png output_subarea ">
     <img src="data:image/png;base64,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
Yn52+wAAAABJRU5ErkJggg==
"/>
    </div>
   </div>
   <div class="output_area">
    <div class="prompt">
    </div>
    <div class="output_png output_subarea ">
     <img src="data:image/png;base64,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
"/>
    </div>
   </div>
   <div class="output_area">
    <div class="prompt">
    </div>
    <div class="output_png output_subarea ">
     <img src="data:image/png;base64,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
YII=
"/>
    </div>
   </div>
   <div class="output_area">
    <div class="prompt">
    </div>
    <div class="output_png output_subarea ">
     <img src="data:image/png;base64,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
LyLiE/8fNYsLlSxhWg0AAAAASUVORK5CYII=
"/>
    </div>
   </div>
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
    Looking at above results we notice that
   </p>
   <ul>
    <li>
     for degree &lt;=2: we are underfitting, the fitted curve has high-bias
    </li>
    <li>
     for degree =3: we are closest to the real curve
    </li>
    <li>
     for dgree &gt; 3: we are high-variance, the curve fits well on central datas but behaves poorly for edge datas.
    </li>
   </ul>
   <p>
    Now let's try it again with regulization $r=10.0$
   </p>
  </div>
 </div>
</div>
<div class="cell border-box-sizing code_cell rendered">
 <div class="input">
  <div class="prompt input_prompt">
   In [48]:
  </div>
  <div class="inner_cell">
   <div class="input_area">
    <div class=" highlight hl-ipython3">
     <pre><span></span><span class="n">clf</span> <span class="o">=</span> <span class="n">linear_regression</span><span class="o">.</span><span class="n">RidgeRegressionModel</span><span class="p">(</span><span class="n">reg</span> <span class="o">=</span> <span class="mi">10</span><span class="p">)</span>
<span class="n">x</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">linspace</span><span class="p">(</span><span class="n">min_x</span><span class="p">,</span> <span class="n">max_x</span><span class="p">,</span> <span class="mi">20</span><span class="p">)</span>

<span class="n">plt</span><span class="o">.</span><span class="n">figure</span><span class="p">(</span><span class="mi">1</span><span class="p">)</span>
<span class="n">deg</span> <span class="o">=</span> <span class="mi">3</span>
<span class="n">clf</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">poly_X</span><span class="p">[:,:</span><span class="n">deg</span><span class="o">+</span><span class="mi">1</span><span class="p">],</span> <span class="n">poly_y</span><span class="p">)</span>
<span class="n">_</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">scatter</span><span class="p">(</span><span class="n">poly_X</span><span class="p">[:,</span><span class="mi">1</span><span class="p">],</span> <span class="n">poly_y</span><span class="p">)</span>
<span class="n">_</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">x</span><span class="p">,</span> <span class="n">clf</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">x</span><span class="p">))</span>
<span class="n">_</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span><span class="s1">'Fit using polynomial degree=</span><span class="si">{}</span><span class="s1">'</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">deg</span><span class="p">))</span>

<span class="n">plt</span><span class="o">.</span><span class="n">figure</span><span class="p">(</span><span class="mi">2</span><span class="p">)</span>
<span class="n">deg</span> <span class="o">=</span> <span class="mi">4</span>
<span class="n">clf</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">poly_X</span><span class="p">[:,:</span><span class="n">deg</span><span class="o">+</span><span class="mi">1</span><span class="p">],</span> <span class="n">poly_y</span><span class="p">)</span>
<span class="n">_</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">scatter</span><span class="p">(</span><span class="n">poly_X</span><span class="p">[:,</span><span class="mi">1</span><span class="p">],</span> <span class="n">poly_y</span><span class="p">)</span>
<span class="n">_</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">x</span><span class="p">,</span> <span class="n">clf</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">x</span><span class="p">))</span>
<span class="n">_</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span><span class="s1">'Fit using polynomial degree=</span><span class="si">{}</span><span class="s1">'</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">deg</span><span class="p">))</span>

<span class="n">plt</span><span class="o">.</span><span class="n">figure</span><span class="p">(</span><span class="mi">3</span><span class="p">)</span>
<span class="n">deg</span> <span class="o">=</span> <span class="mi">5</span>
<span class="n">clf</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">poly_X</span><span class="p">[:,:</span><span class="n">deg</span><span class="o">+</span><span class="mi">1</span><span class="p">],</span> <span class="n">poly_y</span><span class="p">)</span>
<span class="n">_</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">scatter</span><span class="p">(</span><span class="n">poly_X</span><span class="p">[:,</span><span class="mi">1</span><span class="p">],</span> <span class="n">poly_y</span><span class="p">)</span>
<span class="n">_</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">x</span><span class="p">,</span> <span class="n">clf</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">x</span><span class="p">))</span>
<span class="n">_</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span><span class="s1">'Fit using polynomial degree=</span><span class="si">{}</span><span class="s1">'</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">deg</span><span class="p">))</span>

<span class="n">plt</span><span class="o">.</span><span class="n">figure</span><span class="p">(</span><span class="mi">4</span><span class="p">)</span>
<span class="n">deg</span> <span class="o">=</span> <span class="mi">6</span>
<span class="n">clf</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">poly_X</span><span class="p">[:,:</span><span class="n">deg</span><span class="o">+</span><span class="mi">1</span><span class="p">],</span> <span class="n">poly_y</span><span class="p">)</span>
<span class="n">_</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">scatter</span><span class="p">(</span><span class="n">poly_X</span><span class="p">[:,</span><span class="mi">1</span><span class="p">],</span> <span class="n">poly_y</span><span class="p">)</span>
<span class="n">_</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">x</span><span class="p">,</span> <span class="n">clf</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">x</span><span class="p">))</span>
<span class="n">_</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span><span class="s1">'Fit using polynomial degree=</span><span class="si">{}</span><span class="s1">'</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">deg</span><span class="p">))</span>
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
   <div class="output_area">
    <div class="prompt">
    </div>
    <div class="output_png output_subarea ">
     <img src="data:image/png;base64,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
"/>
    </div>
   </div>
   <div class="output_area">
    <div class="prompt">
    </div>
    <div class="output_png output_subarea ">
     <img src="data:image/png;base64,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
"/>
    </div>
   </div>
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
    Using a regularization 10 seems improves the fit for polynomial degree 3 &amp; 5. We can use higher level of $r$ but for $r&gt;10$ it starts has affect on fitting performance. Normally, one can tune the regularization using a validation set.
   </p>
   <h2 id="Conclusition">
    Conclusition
    <a class="anchor-link" href="#Conclusition">
     ¶
    </a>
   </h2>
   <p>
    We have learnt the normal equations for linear regression with/without regularization. The normal equations is not only very effecient but also returns very useful statistics.
   </p>
  </div>
 </div>
</div>
