# rstan {#rstan}

## Resources

- [mc-stan online documentation](https://mc-stan.org/users/interfaces/rstan)

## Description

All of the packages in the previous chapters are running `stan` under the hood. One of the biggest advantages to using `rstan` (or `cmdstanr`) is you get the full power and flexibility of `stan` so you can build models that aren't supported by the other packages. The tradeoff is the syntax is quite different which can be a challenge for those who are only familiar with R.

## Environment Setup


```r
set.seed(123)
options("scipen" = 1, "digits" = 4)

library(tidyverse)
library(datasets)
data(mtcars)
# mean center disp predictor
mtcars$c_disp = mtcars$disp - mean(mtcars$disp)

library(rstan)
library(bayesplot)

# Saves compiled version of model so it only has to be recompiled if the model is changed
rstan_options(auto_write = TRUE)
# Set number of cores
options(mc.cores = parallel::detectCores())
```

## Linear Model

### Define Model

Like the `rethinking` package, `rstan` doesn't have default priors, so I need to explicitly choose them:

\begin{align*}
  mpg &\sim Normal(\mu, \sigma^2) \\
  \mu &= a + b*c\_disp \\
  a &\sim Normal(13.2, 5.3^2) \\
  b &\sim Normal(-0.1, 0.05^2) \\
  \sigma &\sim Exponential(1)
\end{align*}

For a simple linear model there are three sections to the model definition:

1. `data` - This is where the data structures for the known/observed portions of the model (e.g., the number of observations, the number and type of predictors) are defined.

2. `parameters` - This is where the data structures for the parameters to be estimated are defined. For example, the coefficients of the simple linear model belong in this section.

3. `model` - This is where the model (including priors) is defined using the data structures from the previous sections.


```r
# Define model

mdl_code <- '
  data{
    int<lower=1> N;
    vector[N] mpg;
    vector[N] c_disp;
  }
  parameters{
    real a;
    real b;
    real<lower=0.0> sigma;
  }
  model{
    // Likelihood
    mpg ~ normal(a + b * c_disp, sigma);
    // Priors
    a ~ normal(13.2, 5.3);
    b ~ normal(-0.1, 0.05);
    sigma ~ exponential(1);
  }
'
```

A couple of comments about the model definition.  

1. For those only familiar with R, it may seem like a lot of extra "stuff" is going on in the `data` and `parameters` sections.  This is because `stan` is written in C++ which is statically typed, unlike R and Python which are dynamically typed.  Essentially what that means is you must define the type of any variable before you use it.

2. The `lower=` (and `upper=` not shown in this example) options define bounds for a variable. The data is checked against the bounds which can detect errors pre-compilation. Generally, bounds are a good idea but aren't required.

Next, populate the data structures from the `data` section and save in a list.


```r
mdl_data <- list(N = nrow(mtcars),
                 mpg = mtcars$mpg,
                 c_disp = mtcars$c_disp)
```

And this is the call to fit the model.


```r
# Fit model
mdl1 <- stan(model_code=mdl_code, data=mdl_data, model_name="mdl1")
```

### Prior Predictive Distribution

I could manually construct the prior predictive distribution like I did in \@ref(rethinkingprior).  Instead I'll have `stan` generate the prior predictive distribution which will be useful for more complex models. First, create another model with just the `data` and `generated quantities` section.  The `generated quantities` section mirrors the `model` section except it is now drawing samples from the priors without conditioning on the observed data. Also, in the `stan` call set the sampling algorithm for fixed parameters.


```r
# Plot prior predictive distribution
mdl_prior <- '
  data{
    int<lower=1> N;
    vector[N] c_disp;
  }
generated quantities{
  real a_sim = normal_rng(13.2, 5.3);
  real b_sim = normal_rng(-0.1, 0.05);
  real sigma_sim = exponential_rng(1);
  real mpg_sim[N] = normal_rng(a_sim + b_sim * c_disp, sigma_sim);
}
'

N<- 50
D <- seq(min(mtcars$c_disp), max(mtcars$c_disp), length.out = N)
mdl_data_prior <- list(N = N, c_disp=D)

mdl_prior <- stan(model_code=mdl_prior, data=mdl_data_prior, model_name="mdl_prior",
             chains=1, algorithm="Fixed_param")
```


```r
draws <- as.data.frame(mdl_prior) %>%
  head(50)

# Expected value prior predictive distribution
exp_mpg_sim <- apply(draws, 1, function(x) x["a_sim"] + x["b_sim"] * (D)) %>%
  as.data.frame() %>%
  mutate(c_disp = D) %>%
  pivot_longer(-c_disp, names_to="iter", values_to="mpg") 

ggplot() +
  geom_line(data=exp_mpg_sim, mapping=aes(x=c_disp, y=mpg, group=iter), alpha=0.2) 
```

<img src="05_rstan_files/figure-html/mdl1_prior_plot-1.png" width="672" />

### Diagnostics


```r
mcmc_rank_overlay(mdl1, pars=c("a", "b", "sigma"))
```

<img src="05_rstan_files/figure-html/mdl1_trankplot-1.png" width="672" />


```r
print(mdl1)
```

```
## Inference for Stan model: mdl1.
## 4 chains, each with iter=2000; warmup=1000; thin=1; 
## post-warmup draws per chain=1000, total post-warmup draws=4000.
## 
##         mean se_mean   sd   2.5%    25%    50%    75%  97.5% n_eff Rhat
## a      20.01    0.01 0.57  18.88  19.66  20.01  20.39  21.14  2367    1
## b      -0.04    0.00 0.00  -0.05  -0.04  -0.04  -0.04  -0.03  3641    1
## sigma   3.22    0.01 0.41   2.53   2.92   3.18   3.48   4.13  2906    1
## lp__  -57.77    0.03 1.29 -61.11 -58.37 -57.42 -56.81 -56.29  1775    1
## 
## Samples were drawn using NUTS(diag_e) at Tue Dec 29 20:29:42 2020.
## For each parameter, n_eff is a crude measure of effective sample size,
## and Rhat is the potential scale reduction factor on split chains (at 
## convergence, Rhat=1).
```

### Posterior Distribution

The `print` function above displays information about the posterior distributions in addition to `n_eff`.  Alternatively, the `plot` function provides a graphical display of the posterior distributions.


```r
plot(mdl1)
```

```
## ci_level: 0.8 (80% intervals)
```

```
## outer_level: 0.95 (95% intervals)
```

<img src="05_rstan_files/figure-html/mdl1_post_plot-1.png" width="672" />

### Posterior Predictive Distribution

Using the posterior samples, I can plot the expected value of the posterior predictive distribution.


```r
N<- 50
D <- seq(min(mtcars$c_disp), max(mtcars$c_disp), length.out = N)

draws <- as.data.frame(mdl1) %>%
  head(50)

# Expected value posterior predictive distribution
post_pred <- apply(draws, 1, function(x) x["a"] + x["b"]*(D)) %>%
  as.data.frame() %>%
  mutate(c_disp = D) %>%
  pivot_longer(-c_disp, names_to="iter", values_to="mpg") 

ggplot(mapping=aes(x=c_disp, y=mpg)) +
  geom_line(data=post_pred, mapping=aes(group=iter), alpha=0.2) +
  geom_point(data=mtcars)
```

<img src="05_rstan_files/figure-html/mdl1_ppd-1.png" width="672" />

Note that the expected value of the ppd doesn't include $\sigma$. 

An alternative is to have `stan` automatically generate samples from the posterior predictive distribution by adding a `generated quantities` section to the model (similar to what I did for the prior predictive distribution). 


```r
# Define model

mdl_code_ppd <- '
  data{
    int<lower=1> N;
    vector[N] mpg;
    vector[N] c_disp;
  }
  parameters{
    real a;
    real b;
    real<lower=0.0> sigma;
  }
  transformed parameters{
    // Expected value of posterior predictive
    vector[N] Y_hat = a + b * c_disp;
  }
  model{
    // Likelihood
    mpg ~ normal(Y_hat, sigma);
    // Priors
    a ~ normal(13.2, 5.3);
    b ~ normal(-0.1, 0.05);
    sigma ~ exponential(1);
  }
  generated quantities{
    // Posterior Predictive
    real mpg_ppd[N] = normal_rng(Y_hat, sigma);
  }
'

# Fit model
mdl1_ppd <- stan(model_code=mdl_code_ppd, data=mdl_data)

draws <- as.data.frame(mdl1_ppd)

# 95% credible interval for expected value of ppd
Eppd <- draws %>% select(starts_with("Y_hat")) %>%
  apply(2, function(x) quantile(x, probs=c(0.025, 0.5, 0.975))) %>%
  t() %>%
  as.data.frame() %>%
  mutate(c_disp = mtcars$c_disp)

# 95 credible interval for ppd
ppd <- draws %>% select(starts_with("mpg_ppd")) %>%
  apply(2, function(x) quantile(x, probs=c(0.025, 0.5, 0.975))) %>%
  t() %>%
  as.data.frame() %>%
  mutate(c_disp=mdl_data$c_disp)
  
ggplot() +
  geom_line(data=Eppd, mapping=aes(x=c_disp, y=`50%`)) +
  geom_ribbon(data=ppd, mapping=aes(x=c_disp, ymin=`2.5%`, ymax=`97.5%`), 
              alpha=0.5, fill="lightblue") +
  geom_ribbon(data=Eppd, mapping=aes(x=c_disp, ymin=`2.5%`, ymax=`97.5%`), 
              alpha=0.5, fill="dodgerblue") +
  geom_point(data=mtcars, mapping=aes(x=c_disp, y=mpg)) +
  labs(x="c_disp", y="mpg")
```

<img src="05_rstan_files/figure-html/mdl1_ppd_alternate-1.png" width="672" />

The darker blue area is a 95% credible interval for the expected value of the posterior predictive distribution and the lighter blue area is the 95% credible interval for the posterior predictive distribution.

And we can also plot the density overlay using the posterior predictive draws.


```r
yrep <- draws %>%
  head(50) %>%
  select(starts_with("mpg_ppd")) %>%
  as.matrix()

ppc_dens_overlay(mtcars$mpg, yrep)
```

<img src="05_rstan_files/figure-html/unnamed-chunk-1-1.png" width="672" />

## Semi-parametric Model

### Define Model

First, I'll define the splines just as I did with the `rethinking` package.  


```r
library(splines)

num_knots <- 4  # number of interior knots
knot_list <- quantile(mtcars$c_disp, probs=seq(0,1,length.out = num_knots))
B <- bs(mtcars$c_disp, knots=knot_list[-c(1,num_knots)], intercept=TRUE)

df1 <- cbind(c_disp=mtcars$c_disp, B) %>%
  as.data.frame() %>%
  pivot_longer(-c_disp, names_to="spline", values_to="val")

# Plot at smaller intervals so curves are smooth
N<- 50
D <- seq(min(mtcars$c_disp), max(mtcars$c_disp), length.out = N)
B_plot <- bs(D, 
             knots=knot_list[-c(1,num_knots)], 
             intercept=TRUE)

df2 <- cbind(c_disp=D, B_plot) %>%
  as.data.frame() %>%
  pivot_longer(-c_disp, names_to="spline", values_to="val")

ggplot(mapping=aes(x=c_disp, y=val, color=spline)) +
  geom_point(data=df1) +
  geom_line(data=df2, linetype="dashed")
```

<img src="05_rstan_files/figure-html/splines-1.png" width="672" />

Note: the dashed lines are the splines and the points are the values of the spline at the specific values of `mtcars$c_disp`; the points are inputs into the `stan` model.


```r
# Define model

mdl_code <- '
  data{
    int<lower=1> N;
    int<lower=1> num_basis;
    vector[N] mpg;
    matrix[N, num_basis] B;
  }
  parameters{
    real a;
    real<lower=0.0> sigma;
    vector[num_basis] w;
  }
  transformed parameters{
    vector[N] Y_hat = a + B*w;
  }
  model{
    // Likelihood
    mpg ~ normal(Y_hat, sigma);
    // Priors
    a ~ normal(25, 10);
    sigma ~ exponential(1);
    w ~ normal(0, 5);
  }
  generated quantities{
    // Posterior Predictive
    real mpg_ppd[N] = normal_rng(Y_hat, sigma);
  }
'

mdl_data <- list(N=nrow(mtcars),
                 num_basis=ncol(B),
                 B=B,
                 mpg = mtcars$mpg)

# Fit model
mdl1_gam <- stan(model_code=mdl_code, data=mdl_data)
```

### Prior Predictive Distribution


```r
# Define model

mdl2 <- '
  data{
    int<lower=1> N;
    int<lower=1> num_basis;
    //vector[N] mpg;
    matrix[N, num_basis] B;
  }
  generated quantities{
    real a_sim = normal_rng(25, 10);
    real sigma_sim = exponential_rng(1);
    real mpg_sim[N];
    vector[N] Y_hat;
    vector[num_basis] w_sim;
    for (i in 1:num_basis)
      w_sim[i] = normal_rng(0,5);
    Y_hat = a_sim + B * w_sim;
    mpg_sim = normal_rng(Y_hat, sigma_sim);
}
'

mdl_gam_prior <- stan(model_code=mdl2, 
                      data=list(N=N, 
                                num_basis=ncol(B_plot),
                                B=B_plot),
                      chains=1, algorithm="Fixed_param")

draws <- as.data.frame(mdl_gam_prior) %>%
  head(50)

# Expected value prior predictive distribution
exp_mpg_sim <- apply(draws, 1, function(x) {
  x["a_sim"] + B_plot %*% x[grepl("w", names(x))]
}) %>%
  as.data.frame() %>%
  mutate(c_disp = D) %>%
  pivot_longer(-c_disp, names_to="iter", values_to="mpg") 

# 95% interval prior predictive distribution
mpg_sim <- as.data.frame(mdl_gam_prior) %>% select(starts_with("mpg")) %>%
  apply(2, function(x) quantile(x, probs=c(0.025, 0.5, 0.975))) %>%
  t() %>%
  as.data.frame() %>%
  mutate(c_disp = D)

ggplot() +
  geom_line(data=exp_mpg_sim, mapping=aes(x=c_disp, y=mpg, group=iter), alpha=0.2) +
  geom_ribbon(data=mpg_sim, mapping=aes(x=c_disp, ymin=`2.5%`, ymax=`97.5%`), 
              alpha=0.5, fill="lightblue")
```

<img src="05_rstan_files/figure-html/mdl1_gam_prior-1.png" width="672" />

### Diagnostics


```r
# Note that bayesplot methods support tidy selection of parameters
mcmc_rank_overlay(mdl1_gam, pars=vars(a, sigma, starts_with("w[")))
```

<img src="05_rstan_files/figure-html/mdl1_gam_trankplot-1.png" width="672" />


```r
# This is the print.stanfit method and pars must be a character vector
print(mdl1_gam, pars=c("a", "sigma", "w"))
```

```
## Inference for Stan model: 95f5b790b7449ad462398a31917fea97.
## 4 chains, each with iter=2000; warmup=1000; thin=1; 
## post-warmup draws per chain=1000, total post-warmup draws=4000.
## 
##        mean se_mean   sd   2.5%    25%   50%   75% 97.5% n_eff Rhat
## a     20.13    0.06 2.04  16.03  18.80 20.13 21.53 24.14  1111    1
## sigma  2.23    0.01 0.31   1.74   2.01  2.20  2.42  2.92  2387    1
## w[1]  11.97    0.06 2.38   7.28  10.39 12.03 13.50 16.66  1357    1
## w[2]   4.45    0.07 2.61  -0.58   2.65  4.47  6.21  9.59  1522    1
## w[3]  -0.55    0.08 2.88  -6.22  -2.45 -0.56  1.43  4.98  1322    1
## w[4]  -5.58    0.07 3.02 -11.32  -7.62 -5.69 -3.54  0.46  1656    1
## w[5]  -2.33    0.08 3.01  -8.25  -4.36 -2.37 -0.32  3.44  1506    1
## w[6]  -8.76    0.06 2.50 -13.61 -10.46 -8.79 -7.10 -3.75  1555    1
## 
## Samples were drawn using NUTS(diag_e) at Tue Dec 29 20:31:46 2020.
## For each parameter, n_eff is a crude measure of effective sample size,
## and Rhat is the potential scale reduction factor on split chains (at 
## convergence, Rhat=1).
```

### Posterior Distribution


```r
plot(mdl1_gam, pars=c("a", "sigma", "w"))
```

```
## ci_level: 0.8 (80% intervals)
```

```
## outer_level: 0.95 (95% intervals)
```

<img src="05_rstan_files/figure-html/mdl1_gam_post-1.png" width="672" />

### Posterior Predictive Distribution


```r
# 95% credible interval expected value of posterior predictive
Eppd <- as.data.frame(mdl1_gam) %>% select(starts_with("Y_hat")) %>%
  apply(2, function(x) quantile(x, probs=c(0.025, 0.5, 0.975))) %>%
  t() %>%
  as.data.frame() %>%
  mutate(c_disp = mtcars$c_disp)

# 95% credible interval posterior predictive
ppd <- as.data.frame(mdl1_gam) %>% select(starts_with("mpg")) %>%
  apply(2, function(x) quantile(x, probs=c(0.025, 0.5, 0.975))) %>%
  t() %>%
  as.data.frame() %>%
  mutate(c_disp = mtcars$c_disp)

ggplot() +
  geom_line(data=Eppd, mapping=aes(x=c_disp, y=`50%`)) +
  geom_ribbon(data=ppd,
              mapping=aes(x=c_disp, ymin=`2.5%`, ymax=`97.5%`), 
              alpha=0.5, fill="lightblue") +
    geom_ribbon(data=Eppd, mapping=aes(x=c_disp, ymin=`2.5%`, ymax=`97.5%`), 
              alpha=0.5, fill="dodgerblue") +
  geom_point(data=mtcars, mapping=aes(x=c_disp, y=mpg)) +
  labs(x="c_disp", y="mpg")
```

<img src="05_rstan_files/figure-html/mdl1_gam_ppd-1.png" width="672" />

## Semi-parametric Model (Random Walk Prior)

### Define Model

One challenge with splines is choosing the number of knots.  For the previous model, I tried several values for `num_knots` until settling on 4.  However, there is a `stan` case study for splines that uses a novel prior which addresses this issue.  The details are [here](https://mc-stan.org/users/documentation/case-studies/splines_in_stan.html).

For this example, I will set `num_knots=20` and then fit models with and without the random walk prior. This is an example where the `rstan`'s flexibility is an advantage because it would be difficult or impossible to specify the random walk prior in the other packages.


```r
library(splines)

num_knots <- 20  # number of interior knots
knot_list <- quantile(mtcars$c_disp, probs=seq(0,1,length.out = num_knots))
B <- bs(mtcars$c_disp, knots=knot_list[-c(1,num_knots)], intercept=TRUE)

# Define model with smoothing prior
mdl_smooth_code <- '
  data{
    int<lower=1> N;
    int<lower=1> num_basis;
    vector[N] mpg;
    matrix[N, num_basis] B;
  }
  parameters{
    real a;
    real<lower=0.0> sigma;
    vector[num_basis] w_raw;
    real<lower=0.0> tau;
  }
  transformed parameters{
    vector[num_basis] w;
    vector[N] Y_hat;
    w[1] = w_raw[1];
    for (i in 2:num_basis)
      w[i] = w[i-1] + w_raw[i]*tau;
    Y_hat = a + B*w;
  }
  model{
    // Likelihood
    mpg ~ normal(Y_hat, sigma);
    // Priors
    a ~ normal(25, 10);
    sigma ~ exponential(1);
    w_raw ~ normal(0, 1);
    tau ~ normal(0,1);
  }
  generated quantities{
    real mpg_ppd[N] = normal_rng(a + B*w, sigma);
  }
'

mdl_data <- list(N=nrow(mtcars), 
                 num_basis=ncol(B),
                 B=B,
                 mpg = mtcars$mpg)

# Fit model with smoothing prior
mdl2_gam_smooth <- stan(model_code=mdl_smooth_code, data=mdl_data,
                 control=list(adapt_delta=0.99))

# Fit model without smoothing prior
mdl2_gam <- stan(model_code = mdl_code, data=mdl_data,
                 control=list(adapt_delta=0.99))
```

### Diagnostics


```r
mcmc_rank_overlay(mdl2_gam_smooth, pars=vars("a", "sigma", starts_with("w[")))
```

<img src="05_rstan_files/figure-html/gam_rw_trankplot-1.png" width="672" />



```r
print(mdl2_gam_smooth, pars=c("a", "sigma", "w"))
```

```
## Inference for Stan model: 2e3abf0ef6bf631fb50d560f27cb244a.
## 4 chains, each with iter=2000; warmup=1000; thin=1; 
## post-warmup draws per chain=1000, total post-warmup draws=4000.
## 
##         mean se_mean   sd   2.5%    25%    50%    75%  97.5% n_eff Rhat
## a      31.99    0.04 2.01  28.06  30.67  31.99  33.34  36.06  2969    1
## sigma   2.17    0.01 0.34   1.62   1.94   2.14   2.37   2.93  2768    1
## w[1]    0.07    0.02 1.01  -1.89  -0.62   0.06   0.74   2.07  4492    1
## w[2]   -1.03    0.03 1.86  -4.72  -2.26  -1.03   0.20   2.63  3840    1
## w[3]   -2.21    0.04 2.15  -6.43  -3.64  -2.20  -0.85   2.08  3605    1
## w[4]   -3.55    0.04 2.33  -8.21  -5.04  -3.57  -2.02   1.06  4091    1
## w[5]   -5.98    0.04 2.43 -10.70  -7.56  -5.96  -4.42  -1.13  3657    1
## w[6]   -8.31    0.04 2.40 -13.05  -9.91  -8.31  -6.75  -3.48  3051    1
## w[7]   -9.18    0.05 2.52 -14.13 -10.80  -9.18  -7.48  -4.40  3121    1
## w[8]   -9.73    0.04 2.39 -14.45 -11.29  -9.71  -8.18  -5.03  3616    1
## w[9]  -10.50    0.04 2.35 -15.17 -12.04 -10.53  -8.94  -5.88  3692    1
## w[10] -11.98    0.04 2.29 -16.44 -13.50 -12.00 -10.49  -7.40  3315    1
## w[11] -12.98    0.04 2.54 -18.10 -14.64 -13.02 -11.33  -8.05  3220    1
## w[12] -13.29    0.04 2.50 -18.21 -14.97 -13.33 -11.62  -8.30  3468    1
## w[13] -13.82    0.04 2.36 -18.44 -15.37 -13.83 -12.26  -9.27  3582    1
## w[14] -15.57    0.04 2.42 -20.31 -17.16 -15.61 -14.00 -10.64  3033    1
## w[15] -16.31    0.04 2.42 -21.04 -17.94 -16.27 -14.72 -11.54  3157    1
## w[16] -16.59    0.04 2.54 -21.71 -18.30 -16.63 -14.80 -11.67  3436    1
## w[17] -16.64    0.04 2.46 -21.52 -18.24 -16.62 -15.00 -11.88  3308    1
## w[18] -15.92    0.04 2.30 -20.46 -17.47 -15.91 -14.40 -11.39  3939    1
## w[19] -15.81    0.04 2.45 -20.65 -17.41 -15.80 -14.24 -10.92  4103    1
## w[20] -17.39    0.04 2.50 -22.28 -19.04 -17.41 -15.74 -12.33  4038    1
## w[21] -19.53    0.05 2.60 -24.61 -21.27 -19.58 -17.81 -14.32  2924    1
## w[22] -20.52    0.05 2.75 -25.86 -22.35 -20.56 -18.70 -14.91  3069    1
## 
## Samples were drawn using NUTS(diag_e) at Tue Dec 29 20:33:18 2020.
## For each parameter, n_eff is a crude measure of effective sample size,
## and Rhat is the potential scale reduction factor on split chains (at 
## convergence, Rhat=1).
```

### Posterior Distribution


```r
plot(mdl2_gam_smooth, pars=c("a", "sigma", "w"))
plot(mdl2_gam, pars=c("a", "sigma", "w"))
```

<img src="05_rstan_files/figure-html/gam_ppd_compare_coefs-1.png" width="50%" /><img src="05_rstan_files/figure-html/gam_ppd_compare_coefs-2.png" width="50%" />

### Posterior Predictive Distribution


```r
# Draws from regular gam
Epost_pred <- as.data.frame(mdl2_gam) %>% 
  select(starts_with("Y_hat")) %>%
  apply(2, function(x) quantile(x, probs=c(0.055, 0.5, 0.945))) %>%
  t() %>%
  as.data.frame() %>%
  mutate(c_disp = mtcars$c_disp)

# Draws from random walk model
Epost_pred_smooth <- as.data.frame(mdl2_gam_smooth) %>%
  select(starts_with("Y_hat")) %>%
  apply(2, function(x) quantile(x, probs=c(0.055, 0.5, 0.945))) %>%
  t() %>%
  as.data.frame() %>%
  mutate(c_disp = mtcars$c_disp)

rbind(Epost_pred %>% select(c("c_disp", `50%`)) %>% mutate(type="without smoothing prior"),
      Epost_pred_smooth %>% select(c("c_disp", `50%`)) %>% mutate(type="with smoothing prior")) %>%
  ggplot() +
  geom_line( mapping=aes(x=c_disp, y=`50%`, linetype=type), color="blue" ) +
  geom_point(data=mtcars, mapping=aes(x=c_disp, y=mpg)) +
  labs(y="mpg")
```

<img src="05_rstan_files/figure-html/gam_ppd_compare_plot-1.png" width="672" />

The plot above shows that even with a large number of knots (in this case 20), the model with the smoothing prior significantly reduces over-fitting when compared to the model without the smoothing prior.

## Session Info


```r
sessionInfo()
```

```
## R version 4.0.3 (2020-10-10)
## Platform: x86_64-apple-darwin17.0 (64-bit)
## Running under: macOS Big Sur 10.16
## 
## Matrix products: default
## BLAS:   /Library/Frameworks/R.framework/Versions/4.0/Resources/lib/libRblas.dylib
## LAPACK: /Library/Frameworks/R.framework/Versions/4.0/Resources/lib/libRlapack.dylib
## 
## locale:
## [1] en_US.UTF-8/en_US.UTF-8/en_US.UTF-8/C/en_US.UTF-8/en_US.UTF-8
## 
## attached base packages:
## [1] splines   stats     graphics  grDevices datasets  utils     methods  
## [8] base     
## 
## other attached packages:
##  [1] bayesplot_1.7.2      rstan_2.21.2         StanHeaders_2.21.0-6
##  [4] forcats_0.5.0        stringr_1.4.0        dplyr_1.0.2         
##  [7] purrr_0.3.4          readr_1.4.0          tidyr_1.1.2         
## [10] tibble_3.0.4         ggplot2_3.3.2        tidyverse_1.3.0     
## 
## loaded via a namespace (and not attached):
##  [1] httr_1.4.2         jsonlite_1.7.1     modelr_0.1.8       RcppParallel_5.0.2
##  [5] assertthat_0.2.1   stats4_4.0.3       renv_0.12.0        cellranger_1.1.0  
##  [9] yaml_2.2.1         pillar_1.4.7       backports_1.2.0    glue_1.4.2        
## [13] digest_0.6.27      rvest_0.3.6        colorspace_2.0-0   htmltools_0.5.0   
## [17] plyr_1.8.6         pkgconfig_2.0.3    broom_0.7.2        haven_2.3.1       
## [21] bookdown_0.21      scales_1.1.1       processx_3.4.5     farver_2.0.3      
## [25] generics_0.1.0     ellipsis_0.3.1     withr_2.3.0        cli_2.2.0         
## [29] magrittr_2.0.1     crayon_1.3.4       readxl_1.3.1       evaluate_0.14     
## [33] ps_1.4.0           fs_1.5.0           fansi_0.4.1        xml2_1.3.2        
## [37] pkgbuild_1.1.0     tools_4.0.3        loo_2.3.1          prettyunits_1.1.1 
## [41] hms_0.5.3          lifecycle_0.2.0    matrixStats_0.57.0 V8_3.4.0          
## [45] munsell_0.5.0      reprex_0.3.0       callr_3.5.1        compiler_4.0.3    
## [49] rlang_0.4.9        grid_4.0.3         ggridges_0.5.2     rstudioapi_0.13   
## [53] labeling_0.4.2     rmarkdown_2.5      gtable_0.3.0       codetools_0.2-16  
## [57] inline_0.3.17      DBI_1.1.0          curl_4.3           reshape2_1.4.4    
## [61] R6_2.5.0           gridExtra_2.3      lubridate_1.7.9.2  knitr_1.30        
## [65] stringi_1.5.3      parallel_4.0.3     Rcpp_1.0.5         vctrs_0.3.5       
## [69] dbplyr_2.0.0       tidyselect_1.1.0   xfun_0.19
```

