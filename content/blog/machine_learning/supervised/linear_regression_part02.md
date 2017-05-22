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
   In [2]:
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
<span class="kn">from</span> <span class="nn">helpers</span> <span class="k">import</span> <span class="n">linear_regression</span><span class="p">,</span> <span class="n">vis</span>

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
   In [3]:
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

R2-score 0.962

Fitted parameters
coef       fitted    F-score    low 95%    high 95%
-------  --------  ---------  ---------  ----------
theta_0   1.9916     53.1477    1.91815     2.06505
theta_1   3.56031    50.0866    3.42099     3.69964
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
   In [4]:
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

<span class="n">vis</span><span class="o">.</span><span class="n">draw</span><span class="p">(</span><span class="n">poly_X</span><span class="p">[:,</span><span class="mi">1</span><span class="p">],</span> <span class="n">poly_y</span><span class="p">)</span>
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
7GJJ19Q/Pk/Si73+XsvtjN3dv+vup+qfPinpkoi7XSvpJXf/gbu/Lelbkm5LobYX3D13F1ttsa5M
xqz+HN+of/wNSRMpPGcjrYzDbZK+6TVPShoys4tzUFdm3P3fJP2swV2yGLNW6sqMux9392fqH/9C
0guSFp9/PNFxy22wL/IZ1X6aLTYs6cfzPn9VSwcsSy5pt5ntN7MNWRdTl9WYvcfdj9c//omk98Tc
L60xa2UcshirVp/zg/Vf2R83s6t6XFM78vyezHzMzGyNpDFJTy36UqLjlumFNsxst6SLIr602d0f
rd9ns6RTkh7MW20t+JC7V8zsQklPmNmR+swi67p6olFt8z9xdzezuOVYiY9ZgJ6RNOLu/2NmN0ua
knR5xjXlXeZjZmbvkvSIpLvc/ee9fK5Mg93db2z0dTP7tKSPSvptrzeiFqlIunTe55fUb+t5bS0+
RqX+92tm9m3Vfs3uKqQSqCuTMTOzn5rZxe5+vP4r5msxj5H4mMVoZRx6Nlbd1DU/FNz9MTP7WzNb
6e55OB9KFmPWVNZjZmaDqoX6g+6+PeIuiY5bblsxZnaTpHsl3eruJ2Pu9rSky81srZmdI+mTknak
VWMjZvZLZnbe3MeqHQyOPGKfsqzGbIekO+of3yFpyW8XKY9ZK+OwQ9If1VcsXCfpzXntpF5pWpeZ
XWRmVv/4WtXex//d47palcWYNZXlmNWf9+uSXnD3r8bcLdlxy+IocYtHkl9Sred0oP7n7+q3r5b0
2KKjyS+qtpJgc0q1/Z5qPbC3JP1U0q7Ftam2quFg/c/hNGprpa4Mx+zdkv5Z0vcl7ZZ0QdZjFjUO
ku6UdGf9Y5P0tfrXD6nBCqiU6/pcfXwOqraw4INp1FV/7r+XdFzSbP177U9yMmbN6spyzD6k2rGj
Z+fl2c29HDd2ngJAYHLbigEAdIZgB4DAEOwAEBiCHQACQ7ADQGAIdgAIDMEOAIEh2AEgMP8P78I7
p3GWvmAAAAAASUVORK5CYII=
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
   In [15]:
  </div>
  <div class="inner_cell">
   <div class="input_area">
    <div class=" highlight hl-ipython3">
     <pre><span></span><span class="n">clf</span> <span class="o">=</span> <span class="n">linear_regression</span><span class="o">.</span><span class="n">RidgeRegressionModel</span><span class="p">(</span><span class="n">reg</span> <span class="o">=</span> <span class="mf">0.</span><span class="p">)</span>
<span class="n">x</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">linspace</span><span class="p">(</span><span class="n">min_x</span><span class="p">,</span> <span class="n">max_x</span><span class="p">,</span> <span class="mi">20</span><span class="p">)</span>

<span class="k">for</span> <span class="n">deg</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span><span class="mi">7</span><span class="p">):</span>
    <span class="n">clf</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">poly_X</span><span class="p">[:,:</span><span class="n">deg</span><span class="o">+</span><span class="mi">1</span><span class="p">],</span> <span class="n">poly_y</span><span class="p">)</span>
    <span class="n">fig_ax</span> <span class="o">=</span> <span class="n">vis</span><span class="o">.</span><span class="n">draw</span><span class="p">(</span><span class="n">poly_X</span><span class="p">[:,</span><span class="mi">1</span><span class="p">],</span> <span class="n">poly_y</span><span class="p">,</span> <span class="n">title</span><span class="o">=</span><span class="s1">'Fit using polynomial degree=</span><span class="si">{}</span><span class="s1">'</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">deg</span><span class="p">))</span>
    <span class="n">vis</span><span class="o">.</span><span class="n">draw</span><span class="p">(</span><span class="n">x</span><span class="p">,</span> <span class="n">clf</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">x</span><span class="p">),</span> <span class="n">fig_ax</span><span class="o">=</span><span class="n">fig_ax</span><span class="p">,</span> <span class="n">plot_type</span><span class="o">=</span><span class="s1">'plot'</span><span class="p">,</span> <span class="n">colors</span><span class="o">=</span><span class="s1">'r'</span><span class="p">)</span>
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
usNUY3N9MoaZdVJTN7OW9BAwG3g6fevOiHh/rc91282sfT1pM8uDttzMOiJ+rZnPdwNfKsDMuo0v
AlaDrydtZt3Gib0GX0/azLqNE3sNvp60mXUbJ/YafD1pM+s2vrpjDb6etJl1Gyf2Ovh60mbWTVyK
MTMrGCd2M7OCcWI3MysYJ3Yzs4JxYjczKxgndjOzgnFiNzMrGCd2M7OCcWI3MysYJ3Yzs4JxYjcz
KxgndjOzgnFiNzMrGCd2M7OCcWI3MysYJ3Yzs4JxYjczKxgndjOzgnFiNzMrGCd2M7OCcWI3MysY
J3Yzs4JxYjczKxgndjOzgnFiNzMrGCd2M7OCcWI3MysYJ3Yzs4JxYjczKxgndjOzgskksUu6TFJI
mp/F8szMbPqaTuySTgF+B/hl8+GYmVmzZmawjM8AHwG+lcGyWmrdxhHWrN/CttExFg4NsGr5UlYs
W9TpsMzMMtVUYpd0MTASEZskZRRSa6zbOMLlN2xmbPwgACOjY1x+w2YAJ3czK5SaiV3SrcCJFSZd
AfwPkjJMTZJWAisBFi9e3ECI2VizfsvhpF4yNn6QNeu3OLGbWaHUTOwRcX6l9yWdBZwGlFrrJwN3
Szo7Ip6osJy1wFqA4eHhaCbo6dg2OtbQ+2Zm3WrapZiI2AwcX3ot6VFgOCKeyiCuzC0cGmCkQhJf
ODTQgWjMzFqnZ8axr1q+lIH+vgnvDfT3sWr50g5FZGbWGlmMigEgIpZktaxWKNXRPSrGzIous8Te
DVYsW+REbmaF1zOlGDOzXtETLXafmGRmvaTwid0nJplZryl8KWaqE5PMzIqo8IndJyaZWa8pfGKv
dgKST0wys6IqfGL3iUlm1msK33nqE5PMrNcUPrGDT0wys95S+FKMmVmvcWI3MysYJ3Yzs4JxYjcz
KxgndjOzgnFiNzMrGCd2M7OCcWI3MysYJ3Yzs4JxYjczKxgndjOzgnFiNzMrGCd2M7OCcWI3MysY
J3Yzs4JxYjczKxgndjOzgnFiNzMrGCd2M7OCcWI3MysYJ3Yzs4JxYjczKxgndjOzgnFiNzMrGCd2
M7OCcWI3MysYJ3Yzs4JpOrFL+pCkByTdJ+mvswjKzMymb2YzH5b0BuBi4OUR8byk47MJy8zMpqvZ
FvsHgKsj4nmAiHiy+ZDMzKwZzSb2lwK/KelHkv5F0quqzShppaQNkjbs3LmzydWamVk1NUsxkm4F
Tqww6Yr088cBrwFeBVwv6cUREZNnjoi1wFqA4eHho6abmVk2aib2iDi/2jRJHwBuSBP5jyUdAuYD
bpKbmXVIs6WYdcAbACS9FJgFPNVsUGZmNn1NjYoBrgWulfRTYD/w7kplGDMza5+mEntE7AfemVEs
ZmaWAZ95amZWME7sZmYF48RuZlYwTuxmZgXjxG5mVjBO7GZmBdPsOPaOWLdxhDXrt7BtdIyFQwOs
Wr6UFcsWdTosM7Nc6LrEvm7jCJffsJmx8YMAjIyOcfkNmwGc3M3M6MJSzJr1Ww4n9ZKx8YOsWb+l
QxGZmeVL1yX2baNjDb1vZtZrui6xLxwaaOh9M7Ne03WJfdXypQz09014b6C/j1XLl3YoIjOzfOm6
ztNSB6lHxZiZVdZ1iR2S5O5EbmZWWdeVYszMbGpO7GZmBePEbmZWME7sZmYF48RuZlYw6sS9pyXt
BB6b5sfnA09lGE5WHFdjHFdjHFdj8hoXNBfbqRGxoNZMHUnszZC0ISKGOx3HZI6rMY6rMY6rMXmN
C9oTm0sxZmYF48RuZlYw3ZjY13Y6gCocV2McV2McV2PyGhe0Ibauq7GbmdnUurHFbmZmU3BiNzMr
mNwndklrJD0g6V5JN0oaqjLfBZK2SHpI0uo2xPX7ku6TdEhS1aFLkh6VtFnSPZI25Ciudu+v4yR9
X9KD6b/HVpmvLfur1vYr8dl0+r2SXtmqWBqM6/WSdqf75x5Jf9amuK6V9KSkn1aZ3qn9VSuutu8v
SadI+qGkn6X/F/9zhXlau78iItcP4HeAmenzvwL+qsI8fcAvgBcDs4BNwBktjuvfAUuB24HhKeZ7
FJjfxv1VM64O7a+/Blanz1dX+h7btb/q2X7gQuB7gIDXAD9qw3dXT1yvB77Trr+nsvX+FvBK4KdV
prd9f9UZV9v3F3AS8Mr0+Tzg5+3++8p9iz0ibomIA+nLO4GTK8x2NvBQRDwcEfuBrwMXtziu+yMi
d3fQrjOutu+vdPlfSZ9/BVjR4vVNpZ7tvxj4aiTuBIYknZSDuDoiIv4VeGaKWTqxv+qJq+0iYntE
3J0+3wPcD0y+gURL91fuE/sk7yE5yk22CNha9vpxjt6RnRLArZLukrSy08GkOrG/ToiI7enzJ4AT
qszXjv1Vz/Z3Yh/Vu85z0p/v35N0Zotjqlee/w92bH9JWgIsA340aVJL91cu7qAk6VbgxAqTroiI
b6XzXAEcAK7LU1x1eG1EjEg6Hvi+pAfSVkan48rcVHGVv4iIkFRtnG3m+6tg7gYWR8ReSRcC64DT
OxxTnnVsf0maC3wT+HBEPNuOdZbkIrFHxPlTTZd0KfBm4LxIC1STjACnlL0+OX2vpXHVuYyR9N8n
Jd1I8nO7qUSVQVxt31+Sdkg6KSK2pz85n6yyjMz3VwX1bH9L9lGzcZUniIj4rqQvSJofEZ2+4FUn
9ldNndpfkvpJkvp1EXFDhVlaur9yX4qRdAHwEeAtEbGvymw/AU6XdJqkWcDbgJvaFWM1ko6RNK/0
nKQjuGLvfZt1Yn/dBLw7ff5u4KhfFm3cX/Vs/03Au9LRC68BdpeVklqlZlySTpSk9PnZJP+Hn25x
XPXoxP6qqRP7K13f3wH3R8Snq8zW2v3Vzt7i6TyAh0hqUfekjy+m7y8Evls234Ukvc+/IClJtDqu
3yWpiz0P7ADWT46LZHTDpvRxX17i6tD+ehHwA+BB4FbguE7ur0rbD7wfeH/6XMDn0+mbmWLkU5vj
+mC6bzaRDCY4p01x/SOwHRhP/77em5P9VSuutu8v4LUkfUX3luWtC9u5v3xJATOzgsl9KcbMzBrj
xG5mVjBO7GZmBePEbmZWME7sZmYF48RuZlYwTuxmZgXz/wEY4igraXqLkAAAAABJRU5ErkJggg==
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
6y6e3O8Yqp4aAz/6EXTp4q/kDebECvNvPpX+jURnAelur5Ao4edQqEPs0vHCC3689VFH+aYcTZNQ
lOraXBSbxBe0bM8lPzmbg4Y8zIxr/+7H7191lR/aecQRzLnpDuyHVVt8Pow7hVWLP2ClWmvP56tj
s0lNOjmUi0m1ajVrFpx4or871ZNPQmn+jx2W8NSluShRxeVra8SFjfbijSkXwaef+gEADz7I8Jde
4uLSbflv10N5queRTO3QE2cNQr1TWOz3qel+syVmbHAu76+OzSYl/BwK+65JtVq0yN+opGlTeO45
fyclkTTVWnHp1AmuvBKGD+fc8/7BYW89y88/eJ0T57zI4iatearnEbxxcHZumFPbASvZjWvq84ic
TKhJJ4dyMalWUt9955P9qlU+2bctvj92yY6Urw0wo//vf8l1v7iYA4Y8xIUDLuGTFm254M3HKb/1
dD9l8733+r/NkEQ6+WAe0jj8HItklM66dT7Zv/IKPP+8H0ctUkd1Gf8e+zd/Ze8m9H/vRT9w4IMP
/I1aTjgBTjoJjjhCZ551kBd3vEpXMST8nHPOj8R56CE/Udbpp0cdkRSAbFRcxr27mAn3jeewt57j
uA9epUnlSn8h1yGHQL9+0L+/v1NXAzVE1EYJX7yrrvLD5K67zreriuSB+LOEhhvWc/CXHzK84ed0
nfUmvPuuX7F1azj6aH8AOProwpvyY80aePNNmDzZX8hWx8kKlfAF7rvP/wGddZa//6yGX0qeqHVa
hy+/9MOHJ070j+XL/Qr77utr/v36+Un+6tsos40b4b33fIKfPNnPSLtmjT+zOftsuOuuOm1WCT/P
5Lzt/vnn/Vwnffv6Ox3Vt38MKWi7XvYsiTKPAQvj55TfuNHfo+H5533yf/NNfze2xo19f1T1AaBT
pxxEXgeffLI5wU+Z4m8uBP7uY337+sdPfuK/Tx2lmvA1LDPLho+bzX+mLWKDc5SYceqB7enTscUW
p6/VVxsC4ST9mTN9B1ivXvDEE0r2knfSGqLcoAHst59/XHGFH9UzZcrmA8C4cX69Vq2gc2fYY4/N
P6ufN20a8jeKsXy5j686yS9c6N9v2xYGDPAJ/ogjYJddchdTILQavpmNAAYA64CPgd8651bU9Jn6
XsMfPm42j0z9fKv3t29Uwg/rtr68e6tZCbPh88/hoIP8KeLUqdCmTXa3L5IFWZsF1Dl/t64XXoA5
c+Cjj2DBAj+9c6yWLZMfDJo1q72cqipYvXrLxw8/bH6+apXvd5g82Ve4wB9kfvrTzbX4Ll1Ca1bN
hxr+JGBYcF/bvwDDgEtDLC9y/5m2KOH7iZI9UONd7+tkxQr42c/8H+AbbyjZS97K2n1/zaBbN/+I
tXq1b0pZsGDzQWDBAnj5ZX8VcKwdd/SJf9ttt07q1Y/162uPpbQUDj0UbrjBJ/j99ot0+uhEQovG
OfdCzMupwIlhlZUvNqR5tlSSzaP92rV+ytqPPvKnuj3y/+47UtzCmgUUgO22823kPXtuvayycvPB
IPaAUFUFLVpAu3b+8+k+OnXyN4zJY7k6/PwOeDzRAjMbDAwG6NChQ47CCUf13BypSvcAkZRzfiRO
de3liCw3E4kUkrIyXyEqwkpRRgnfzCYDOydYdIVz7ulgnSuA9cDoRNtwzo0ERoJvw88knijEjr7Z
trQBlVVbf4Wa2vCzYtQoGD3aj7f/zW+ys00RKTgZJXznXN+alpvZmcCxwJEun8Z/Zkl8x1Nl1UY/
OZHBRkfSUTqQxTl0li6FP/7RD+u6/PLMtyciBSu0Jh0z6w/8GTjMObc6rHKilGia2I1A26aJR99k
fRy+c3D++f7CjXvv1SXoIlKjMNvw/wlsA0wy3zk51Tl3bojl5Vw689uH0kH15JN+DPJf/+pHGYiI
1CDMUTp7hLXtfBHp/PZffw1DhvihXxdfHH55IlLvqQ0gA5HOb3/xxfDNN3D//Xk31ldE8pMyRQay
dvFIup57zg+/vPJK2GuvcMsSkYKhydPqm++/9+OHmzTxl3Jvs03UEYlIxPJhagUJw2WXQUWF77BV
sheRNKgNvz555RU/X/ZFF8GBB0YdjYjUM0r49UVlpb9Bwm67+cmZRETSpCad+uLqq/0ET1Om+Ima
RETSpBp+ffDOO3DrrXDOOX5+bRGROlDCz3fr1vmZMHfeGUaMiDoaEanH1KST7265BWbPhvHjc3ub
NhEpOKrh57O5c30H7amn+nthiohkQAk/X23Y4JtymjaF22+POhoRKQBq0slXt98O06bBo49Cq1ZR
RyMiBUA1/Hz08ccwfLhvxjnllKijEZECoYSfb5zzwy9LS/1Vtdm80bmIFDU16eSbf/8bXnoJRo6E
tiHPuikiRUU1/HyyeDEMHeovrjr77KijEZECE3rCN7M/mZkzs5Zhl1WvOQfnnQdVVb6Wr6YcEcmy
UJt0zKw9cDTweZjlZGrczIrc38Qk3mOPwYQJcNttsPvuuS1bRIpC2DX8vwN/BvLnLitxxs2sYNiY
2VSsqMQBFSsqGTZmNuNmVuQuiG++gQsv9FMeX3hh7soVkaISWsI3s+OACufcrFrWG2xm081s+rJl
y8IKJ6kRE+dTWbVhi/cqqzYwYuL83AVx663+puT33AMlJbWvLyJSBxk16ZjZZGDnBIuuAC7HN+fU
yDk3EhgJ/haHmcRTF0tWVKb1ftYtXw533AGDBsHee+emTBEpShklfOdc30Tvm1kvYFdglvnOx3bA
u2Z2gHPui0zKzLY2zcqoSJDc2zQry00Af/sb/PADXHVVbsoTkaIVSqetc2420Lr6tZl9CvRxzi0P
o7y6Gjezgh/Wrt/q/bLSEob26xp+AF99Bf/4h58crXv38MsTkaJWtBdeVXfWxrffN9+ulKsH9MjN
KJ0RI2DNGtXuRSQncpLwnXOdclFOOhJ11gJs16hhbpL9F1/AnXfCr38NXXNwNiEiRa9or7SNvLP2
r3/1d7O68srclCciRa9oE36yTtmcdNYuXeonRjvtNOjcOfzyREQo4oQ/tF9Xykq3HPOes87aW27x
UygMHx5+WSIigaLttK1up8/5lAoVFf4CqzPP1BQKIpJTRZvwwSf9nM+Zc/PN/vaFqt2LSI4VbZNO
JBYt8jNh/u530KlT1NGISJFRws+lm27y0yBfcUXUkYhIEVLCz5XPPoP77vM3NunQIepoRKQIKeHn
yo03+puaXH551JGISJFSws+FhQvhgQdg8GBo1y7qaESkSCnh58INN/h57ocNizoSESliSvhhW7AA
HnwQzj0X2rSJOhoRKWJK+GG74QYoLYVLL406EhEpckr4YfrwQ3j4YTj/fNhll6ijEZEip4Qfpuuv
h222gT//OepIRESU8EPzwQfw6KMwZAjstFPU0YiIKOGH5rrroKwMhg6NOhIRESDkhG9mF5jZB2Y2
18z+GmZZeWXuXHjsMbjgAmjVKupoRESAEGfLNLOfAscBezvn1ppZ69o+UzCuuw623x4uuSTqSERE
Ngmzhn8ecItzbi2Ac+6rEMvKH7NnQ3k5/OEPsOOOUUcjIrJJmAm/C/BjM5tmZq+Y2f6JVjKzwWY2
3cymL1u2LMRwcuTaa6FJE/jjH6OORERkCxk16ZjZZGDnBIuuCLbdAjgI2B8oN7PdnHMudkXn3Ehg
JECfPn1c/Ibqlffeg6eegquughYtoo5GRGQLGSV851zfZMvM7DxgTJDg3zazjUBLoACq8Ulcey00
bQoXXxyL2SXTAAAMxElEQVR1JCIiWwmzSWcc8FMAM+sCNAKWh1hetGbMgHHjfFNOs2ZRRyMispUw
72l7P3C/mc0B1gFnxDfnFJRrroHmzX1nrYhIHgot4Tvn1gG/CWv7eWXWLJgwwU+U1rRp1NGIiCQU
Zg0/58bNrGDExPksWVFJm2ZlDO3XlYG924Zf8N13w7bbwnnnhV+WiEgdFUzCHzezgmFjZlNZtQGA
ihWVDBszGyDcpL9yJTzyCJx8skbmiEheK5i5dEZMnL8p2VerrNrAiInzwy340Udh1Sp/gxMRkTxW
MAl/yYrKtN7PCufgrrtg773hwAPDK0dEJAsKJuG3aVaW1vtZ8fbbvsP23HPBLLxyRESyoGAS/tB+
XSkrLdnivbLSEob26xpeoXffDTvsAL/+dXhliIhkScF02lZ3zOZslM633/opkM84Axo3DqcMEZEs
KpiEDz7p52QYJsBDD8GaNRqKKSL1RsE06eSUc74556CDfIetiEg9UFA1/Jx59VV/z9pRo6KOREQk
Zarh18Xdd/sJ0gYNijoSEZGUKeGn66uv/Jz3Z57pb1IuIlJPKOGn64EHoKoKfv/7qCMREUmLEn46
Nm6Ee+6Bww+Hbt2ijkZEJC1K+OmYNAkWLtS8OSJSLynhp+Ouu6BVKzj++KgjERFJmxJ+qhYvhmee
gbPOgkaNoo5GRCRtoSV8M9vHzKaa2XtmNt3MDgirrJy4915/wdU550QdiYhInYRZw/8rcK1zbh/g
quB1/bR+Pfz739CvH+y2W9TRiIjUSZgJ3wFNgudNgSUhlhWuCRNgyRJ11opIvWbOuXA2bLYnMBEw
/IHlEOfcZwnWGwwMBujQocN+n3221SrR698f5syBTz+FhpqNQkTyi5nNcM71qW29jGr4ZjbZzOYk
eBwHnAdc7JxrD1wM3JdoG865kc65Ps65Pq1atcoknHB88glMnOjb7pXsRaQeyyiDOef6JltmZg8B
fwhePgHcm0lZkRk5EkpK4Oyzo45ERCQjYbbhLwEOC54fAXwUYlnhWLsW7r8fBgyAtjmaZ19EJCRh
tlGcA9xuZg2BNQTt9PXK2LGwbJk6a0WkIISW8J1zrwP7hbX9nLj7bth1VzjqqKgjERHJmK60TWbe
PHjlFT8rZgPtJhGp/5TJkrnnHigthd/+NupIRESyQgk/kdWr4cEH4cQToXXrqKMREckKJfxEysth
xQp11opIQVHCT+Tuu2HPPeHHP446EhGRrFHCjzdzJkyb5mv3ZlFHIyKSNUr48e65x9+c/LTToo5E
RCSrlPBjrVwJo0fDKadA8+ZRRyMiklVK+LFGj4ZVq9RZKyIFSQm/mnO+s7Z3b9h//6ijERHJOs33
W23aNJg1y7fhq7NWRAqQavjV7r4bGjeGU0+NOhIRkVAo4QN8+y08/jj85jc+6YuIFCAlfIAxY2DN
GjjrrKgjEREJjRI++KkUdt8d9t036khEREKjhL98Obz4Ipx0kjprRaSgKeGPGwcbNsCgQVFHIiIS
qowSvpmdZGZzzWyjmfWJWzbMzBaY2Xwz65dZmCEqL4c99oB99ok6EhGRUGVaw58DnAC8GvummXUH
TgF6AP2Bf5lZSYZlZd/y5TBlippzRKQoZJTwnXPznHPzEyw6DnjMObfWObcQWAAckElZoRg7Vs05
IlI0wmrDbwssinm9OHgvv5SXQ+fOsPfeUUciIhK6WqdWMLPJwM4JFl3hnHs60wDMbDAwGKBDhw6Z
bi51y5bBSy/BpZeqOUdEikKtCd8517cO260A2se8bhe8l2j7I4GRAH369HF1KKtu1JwjIkUmrCad
8cApZraNme0KdAbeDqmsuqluztlrr6gjERHJiUyHZR5vZouBg4FnzWwigHNuLlAO/A94Hvg/59yG
TIPNmurmnEGD1JwjIkUjo+mRnXNjgbFJlt0I3JjJ9kMzZgxs3KjmHBEpKsV5pW15OXTpAr16RR2J
iEjOFF/C/+orePllNeeISNEpvoSv5hwRKVLFl/DLy6FrV+jZM+pIRERyqrgS/pdfwiuvqDlHRIpS
cSV8NeeISBErroT/xBPQrRv06BF1JCIiOVc8Cf+LL9ScIyJFrXgSvppzRKTIFU/Cf+IJ2HNPNeeI
SNEqjoQf25wjIlKkiiPhP/UUOOdvZSgiUqSKI+E/8QR0767mHBEpaoWf8JcuhVdfVXOOiBS9wk/4
as4REQGKIeE/8YRvyunePepIREQiVdgJf+lSeO01NeeIiFDoCV/NOSIim2R6T9uTzGyumW00sz4x
7x9lZjPMbHbw84jMQ62D8nI/DfKee0ZSvIhIPsm0hj8HOAF4Ne795cAA51wv4Azg4QzLSd+SJfD6
62rOEREJZHoT83kAFjcZmXNuZszLuUCZmW3jnFubSXlpUXOOiMgWctGG/0vg3WTJ3swGm9l0M5u+
bNmy7JVaXu5vUt6tW/a2KSJSj9Wa8M1sspnNSfA4LoXP9gD+Avw+2TrOuZHOuT7OuT6tWrVKL/pk
KirUnCMiEqfWJh3nXN+6bNjM2gFjgdOdcx/XZRt19tRT/qeac0RENgmlScfMmgHPApc5594Io4wa
lZfDXnv5m5WLiAiQ+bDM481sMXAw8KyZTQwWDQH2AK4ys/eCR+sMY03N4sXwxhuq3YuIxMl0lM5Y
fLNN/Ps3ADdksu06U3OOiEhChXelbXk57L23mnNEROIUVsJftAjefFO1exGRBAor4as5R0QkqcJK
+OXlsM8+0KVL1JGIiOSdwkn4ixbBW2+pdi8ikkThJPwnn/Q/lfBFRBIqnIRfXg69e0PnzlFHIiKS
lwoj4X/+OUydqtq9iEgNCiPhr1oFv/iFEr6ISA0yutI2b3TvDk8/HXUUIiJ5rTBq+CIiUislfBGR
IqGELyJSJJTwRUSKhBK+iEiRUMIXESkSSvgiIkVCCV9EpEiYcy7qGDYxs2XAZxlsoiWwPEvhZJPi
So/iSo/iSk8hxtXROdeqtpXyKuFnysymO+f6RB1HPMWVHsWVHsWVnmKOS006IiJFQglfRKRIFFrC
Hxl1AEkorvQorvQorvQUbVwF1YYvIiLJFVoNX0REklDCFxEpEvU24ZvZCDP7wMzeN7OxZtYsyXr9
zWy+mS0ws8tyFNtJZjbXzDaaWdJhVmb2qZnNNrP3zGx6HsWV031mZi3MbJKZfRT8bJ5kvdD3V23f
3bw7guXvm9m+YcRRh7gON7Pvgn3znpldlaO47jezr8xsTpLlUe2v2uKKan+1N7OXzOx/wf/iHxKs
E94+c87VywdwNNAweP4X4C8J1ikBPgZ2AxoBs4DuOYhtT6Ar8DLQp4b1PgVa5nCf1RpXFPsM+Ctw
WfD8skS/y1zsr1S+O3AM8F/AgIOAaTn4vaUS1+HAhFz9LcWU+xNgX2BOkuU5318pxhXV/toF2Dd4
3hj4MJd/Y/W2hu+ce8E5tz54ORVol2C1A4AFzrlPnHPrgMeA43IQ2zzn3Pywy0lXinFFsc+OAx4M
nj8IDAy5vGRS+e7HAQ85byrQzMx2yYO4IuGcexX4poZVothfqcQVCefcUufcu8HzlcA8oG3caqHt
s3qb8OP8Dn9EjNcWWBTzejFb79woOWCymc0ws8FRBxOIYp/t5JxbGjz/AtgpyXph769UvnsU+yfV
Mg8JmgD+a2Y9Qo4pVfn8Pxjp/jKzTkBvYFrcotD2WV7fxNzMJgM7J1h0hXPu6WCdK4D1wOh8iy0F
P3LOVZhZa2CSmX0Q1Eyijivraoor9oVzzplZsrHCWd9fBeRdoINzbpWZHQOMAzpHHFM+i3R/mdkO
wFPARc6573NVbl4nfOdc35qWm9mZwLHAkS5o/IpTAbSPed0ueC/02FLcRkXw8yszG4s/dc8ogWUh
rlD2WU1xmdmXZraLc25pcOr6VZJtZH1/xUnlu4f2N5VJXLFJwzn3nJn9y8xaOueiniQsiv1Vqyj3
l5mV4pP9aOfcmASrhLbP6m2Tjpn1B/4M/MI5tzrJau8Anc1sVzNrBJwCjM9VjDUxs+3NrHH1c3wn
dMIRBTkWxT4bD5wRPD8D2OpMJEf7K5XvPh44PRhJcRDwXUxzVFhqjcvMdjYzC54fgP/f/jrkuFIR
xf6qVVT7KyjzPmCec+62JKuFt89y3UudrQewAN/O9V7wuDt4vw3wXMx6x+B7wj/GN2vkIrbj8e1u
a4EvgYnxseFHXMwKHnNzEVsqcUWxz4AdgReBj4DJQIuo9lei7w6cC5wbPDfgzmD5bGoYhZXjuIYE
+2UWfhDDITmK6z/AUqAq+Ns6K0/2V21xRbW/foTvi3o/Jncdk6t9pqkVRESKRL1t0hERkfQo4YuI
FAklfBGRIqGELyJSJJTwRUSKhBK+iEiRUMIXESkS/x9/1AYiM5hmvgAAAABJRU5ErkJggg==
"/>
    </div>
   </div>
   <div class="output_area">
    <div class="prompt">
    </div>
    <div class="output_png output_subarea ">
     <img src="data:image/png;base64,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
2Ob2HRoAAAAASUVORK5CYII=
"/>
    </div>
   </div>
   <div class="output_area">
    <div class="prompt">
    </div>
    <div class="output_png output_subarea ">
     <img src="data:image/png;base64,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
AElFTkSuQmCC
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
   In [16]:
  </div>
  <div class="inner_cell">
   <div class="input_area">
    <div class=" highlight hl-ipython3">
     <pre><span></span><span class="n">clf</span> <span class="o">=</span> <span class="n">linear_regression</span><span class="o">.</span><span class="n">RidgeRegressionModel</span><span class="p">(</span><span class="n">reg</span> <span class="o">=</span> <span class="mi">10</span><span class="p">)</span>
<span class="n">x</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">linspace</span><span class="p">(</span><span class="n">min_x</span><span class="p">,</span> <span class="n">max_x</span><span class="p">,</span> <span class="mi">20</span><span class="p">)</span>

<span class="k">for</span> <span class="n">deg</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="mi">3</span><span class="p">,</span><span class="mi">7</span><span class="p">):</span>
    <span class="n">clf</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">poly_X</span><span class="p">[:,:</span><span class="n">deg</span><span class="o">+</span><span class="mi">1</span><span class="p">],</span> <span class="n">poly_y</span><span class="p">)</span>
    <span class="n">fig_ax</span> <span class="o">=</span> <span class="n">vis</span><span class="o">.</span><span class="n">draw</span><span class="p">(</span><span class="n">poly_X</span><span class="p">[:,</span><span class="mi">1</span><span class="p">],</span> <span class="n">poly_y</span><span class="p">,</span> <span class="n">title</span><span class="o">=</span><span class="s1">'Fit using polynomial degree=</span><span class="si">{}</span><span class="s1">'</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">deg</span><span class="p">))</span>
    <span class="n">vis</span><span class="o">.</span><span class="n">draw</span><span class="p">(</span><span class="n">x</span><span class="p">,</span> <span class="n">clf</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">x</span><span class="p">),</span> <span class="n">fig_ax</span><span class="o">=</span><span class="n">fig_ax</span><span class="p">,</span> <span class="n">plot_type</span><span class="o">=</span><span class="s1">'plot'</span><span class="p">,</span> <span class="n">colors</span><span class="o">=</span><span class="s1">'r'</span><span class="p">)</span>
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
e/jhAAAAAElFTkSuQmCC
"/>
    </div>
   </div>
   <div class="output_area">
    <div class="prompt">
    </div>
    <div class="output_png output_subarea ">
     <img src="data:image/png;base64,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
VyoCqjQ77QT/+hf87nfw4otw4IGhfZHnnuuwWNzZdyGKKeJq+WTzY48rqkkc4POxjWxkeg8kUUSl
g3+Fe+WVULtnyJDwtP4ee4QG3VIO/tD51Xex6TmASvbee+Hpwp//PNyAOvVU+NGPoLExtv57OT/o
VC6XzeniaehVx5p1GzvciC73/Sk5coeHHgq1embMCO8/97lwk/fjH48tfu3syff/+eLonH8vehBM
OnrjjdCa6DXXQO/e4UzlrLOY+fTbZXVA7e7ylaDKLdFJinXr4M9/Dgf+BQtghx1CT36nnQbDhsU+
WZ6YVhPzJDvk56RBCUDSe+qpcPCfNQt22QV+8pPQrlCVdTRTzrrjlVnVeO210EfHtGmwYkWoffft
b8NXvhJOrIhvIgUnbTXldDJ92jyOngSW9D78YfjrX0M7JAMHhuZnx44ND5BJ1nJ5jiJOOZUPS+SR
R+DLXw6tdP7kJ7D//uF/ZskSOOWU9w/+EF89ONODPxT3eRElgG4kbwecgw8OvQ3dfDOsWgVHHBFu
Vi1enN+AK1ihmnsoRO0s2QobNoT/j/33h/32C82vn3EGPPtsuII+7LC0Zfz5+DsV83mRvCQAMzvK
zJ4xs6VmNiXNfDOzK6P5i8xsbD62W03yfsCpqQnPCTz9dOhy8uGHYfRoOOkkaFGbNV0p1Jl6LrWz
CnFFUjXa2mDRIvjDH8KBfvjwUJX67bfhyivD/8SvfgW7797panI9eBf7eZGcE4CZ1QJXA0cDo4AT
zGxUymJHAyOiYTLw21y3W23iDjgX/+3J3Fa8zTahs5mlS+G734U//jG0UnjyyeFG14oVua2/QhXq
TL2r6q5x1ABdFtra4IknwsOSp50WzvC32w722is8uXv99TBmDMyeHU6Qzjwz486X0v396mrjm31o
qK8rWtXldPLxINi+wFJ3fx7AzG4BxgP/TlpmPHBD1A/wfDNrMLOB7r4sD9uvCnEHlnfWtjFzQUvu
P5p+/cJDY6efHmoM3XZb+AeBcGVw+OHhsvegg6BXr9y2VQG6ejAtWTa1eo4dMxhra+Oqu5bwxsq1
DOrbk28fMpRxg+tCByKbNoWhvb3D6/W/f4AhreupcafGHTdYX9uTG295m2N3/mRI9ImhRxU9/7lh
Qyirf+yxzcOiRWE6hAP72LEhEey9dxhGjNjqihHHjhlM80tvc/PDr9DuTq0ZX9xnCAA3zX+5Q/XP
+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
VUoJQESkSv1/WmYbsoIsnaQAAAAASUVORK5CYII=
"/>
    </div>
   </div>
   <div class="output_area">
    <div class="prompt">
    </div>
    <div class="output_png output_subarea ">
     <img src="data:image/png;base64,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
VVVVtFoVmOGA+cuaCk7K7W1b0Y93kbS2+uGMf/QjfxHVT34Cv/99zhdRFaNhvlRtPaEmfDOrBm4B
DgcWAW+Y2SPOuXfD3G+lSNcrxAHjp35C3eDe5fHP/dVXMGsWvP02fPHFxuS9YkX6pL5ypb9wpYN5
LuXxBquipaqaDVXVcF0X/8NQUwOdO0NNDStbq/h8bStrqILOXejftwd9e/fwyxO3YP0/vrOUxg2w
tqYza2q60lTTlTU1XenaswdMaobu3Te7PTZ3Bb94/EPWBf0uGhqbuPjBtwD1+8/JJ5/As8/623PP
+e9u//7w9NN5X0TVs7Ym7QV7ucq3basQYZfw9wXmOuc+BjCz+4DjASX8HGQqOTjIuStlUX32GcyY
ATNnbvw7d+6mpeyuXWHLLaFnT/93yy1h6NCN95OfT711Svo6mm1+P91zyfdbW/2PSUtL/rfk123Y
sPEWPL7l2fdZvWYd1a0tdGptpdq10Km1hS07V3H63v2hudmvv349Cz9vZPb8L6jqtIGalmZqWjaw
4POvqF67lt6dDdav3+R2dNM61q9pomvzOmpaU7p9PpD+ozgmuK2rrmFNTVdWd+7KV126s+reLWHk
jr5/eLZb9+6bHsdKtmKF72kzebJP8nPm+Of79YOjjoJRo+C44/z3MweJs6tMVTlVBpmq73vV1tC9
S6dIzszCTvgDgIVJjxcB+yWvYGajgdEAgwYNCjmcjqWtXiGh9u9ubfWJPDmxz5gBn3++cZ2hQ2Hk
SPje9/zfvfaCbbf1JdcKNOCwzA2rpPyznnrd8zTsmXudfBfgySCBLF22kh26GRcc0J8jh24Jq1fz
8oz5TKyfQ1PjSvrXtHLsTj15/NW5dGteS23zWro1r6N7cxM91q2hZ9NXPpktXw7LlvkflUxqamCr
rTb+APTt64cKyHTbemtfzdURNDfDa69tLMW//rr/8e7eHQ45BM491/ep3333vH/0Js1o4OIH32qz
QbZnbQ1H79lvs0b+2ppqrjxuWGRnYZF3y3TOjQXGgm+0jTicsnLxEbty4f0z0/YKKWqd35o18NBD
/p9ixgx46y1Yvdov69QJhg2DI4/0iX3ECJ/ce/Uq3v47gGxdOZO1p04+U9vKpBkNXDZ/CU3bDodt
/XP3VFfTtN+eGbc1P9H4l2gvWb7cT86xfHnbt3nzfJJcunRje0oyM+jTp+0fhF69Nt622mrzM7ew
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
MChoqwAAAABJRU5ErkJggg==
"/>
    </div>
   </div>
   <div class="output_area">
    <div class="prompt">
    </div>
    <div class="output_png output_subarea ">
     <img src="data:image/png;base64,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
TkSuQmCC
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
