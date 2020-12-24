---
output:
  html_document: default
  pdf_document: default
---
# brms {#brms}

## Resources

- [Overview of brms](https://cran.r-project.org/web/packages/brms/vignettes/brms_overview.pdf)

- [Solomon Kurz's translation of _Statistical Rethinking_](https://bookdown.org/content/4857/)

## Description


## Environment Setup


```r
set.seed(123)
options("scipen" = 1, "digits" = 4)
knitr::opts_chunk$set(message=FALSE)

library(tidyverse)
library(datasets)
data(mtcars)

library(brms)
library(bayesplot)


# Set number of cores
options(mc.cores = parallel::detectCores())
```

## Linear Model

### Define Model

The `brms` package default priors are improper flat priors over the real line. However, there is a strong case to be made against this type of non-informative prior ^[https://mc-stan.org/users/documentation/case-studies/weakly_informative_shapes.html]. So I'll proceed directly to specifying priors based on the EPA data.

\begin{align*}
  mpg &\sim Normal(\mu, \sigma^2) \\
  \mu &= a + b*disp \\
  a &\sim Normal(13.2, 5.3^2) \\
  b &\sim Normal(-0.1, 0.05^2) \\
  \sigma &\sim Exponential(1)
\end{align*}


```r
mdl1 <- brm(mpg ~ disp, data=mtcars, family=gaussian(), 
            prior=c(set_prior("normal(-0.1, 0.05)", class="b", coef = "disp"),
                    set_prior("normal(13.2, 5.3)", class="Intercept")))
```


```r
stancode(mdl1)
```

```
## // generated with brms 2.14.4
## functions {
## }
## data {
##   int<lower=1> N;  // total number of observations
##   vector[N] Y;  // response variable
##   int<lower=1> K;  // number of population-level effects
##   matrix[N, K] X;  // population-level design matrix
##   int prior_only;  // should the likelihood be ignored?
## }
## transformed data {
##   int Kc = K - 1;
##   matrix[N, Kc] Xc;  // centered version of X without an intercept
##   vector[Kc] means_X;  // column means of X before centering
##   for (i in 2:K) {
##     means_X[i - 1] = mean(X[, i]);
##     Xc[, i - 1] = X[, i] - means_X[i - 1];
##   }
## }
## parameters {
##   vector[Kc] b;  // population-level effects
##   real Intercept;  // temporary intercept for centered predictors
##   real<lower=0> sigma;  // residual SD
## }
## transformed parameters {
## }
## model {
##   // likelihood including all constants
##   if (!prior_only) {
##     target += normal_id_glm_lpdf(Y | Xc, Intercept, b, sigma);
##   }
##   // priors including all constants
##   target += normal_lpdf(b[1] | -0.1, 0.05);
##   target += normal_lpdf(Intercept | 13.2, 5.3);
##   target += student_t_lpdf(sigma | 3, 0, 5.4)
##     - 1 * student_t_lccdf(0 | 3, 0, 5.4);
## }
## generated quantities {
##   // actual population-level intercept
##   real b_Intercept = Intercept - dot_product(means_X, b);
## }
```


### Prior Predictive Distribution


```r
D <- seq(min(mtcars$disp), max(mtcars$disp))

mdl1_prior <- brm(mpg ~ disp, data=mtcars, family=gaussian(), 
            prior=c(set_prior("normal(-0.1, 0.05)", class="b", coef = "disp"),
                    set_prior("normal(13.2, 5.3)", class="Intercept")),
            sample_prior="only")

# Samples from posterior predictive distribution
ppd <- as.data.frame(predict(mdl1_prior, newdata=data.frame(disp=D)))
# Samples from expected value of posterior predictive distribution
eppd <- posterior_epred(mdl1_prior, newdata=data.frame(disp=D), summary=FALSE) %>%
  head(50) %>%
  t() %>%
  as.data.frame() %>%
  mutate(disp=D) %>%
  pivot_longer(-disp, names_to="iter", values_to="mpg")


ggplot() +
  geom_ribbon(data=ppd, mapping=aes(x=D, ymin=Q2.5, ymax=Q97.5), alpha=0.5, fill="lightblue") +
  geom_line(data=eppd, mapping=aes(x=disp, y=mpg, group=iter), alpha=0.2) 
```

<img src="04_brms_files/figure-html/mdl1_prior-1.png" width="672" />


### Diagnostics


```r
mcmc_rank_overlay(mdl1, pars=c("b_Intercept", "b_disp", "sigma"))
```

<img src="04_brms_files/figure-html/mdl1_trankplot-1.png" width="672" />


```r
summary(mdl1)
```

```
##  Family: gaussian 
##   Links: mu = identity; sigma = identity 
## Formula: mpg ~ disp 
##    Data: mtcars (Number of observations: 32) 
## Samples: 4 chains, each with iter = 2000; warmup = 1000; thin = 1;
##          total post-warmup samples = 4000
## 
## Population-Level Effects: 
##           Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS Tail_ESS
## Intercept    29.62      1.28    27.08    32.16 1.00     3956     2991
## disp         -0.04      0.00    -0.05    -0.03 1.00     4111     2878
## 
## Family Specific Parameters: 
##       Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS Tail_ESS
## sigma     3.37      0.46     2.60     4.41 1.00     2946     2875
## 
## Samples were drawn using sampling(NUTS). For each parameter, Bulk_ESS
## and Tail_ESS are effective sample size measures, and Rhat is the potential
## scale reduction factor on split chains (at convergence, Rhat = 1).
```

### Posterior Distribution


```r
fixef(mdl1)
```

```
##           Estimate Est.Error     Q2.5    Q97.5
## Intercept 29.61988  1.279003 27.08427 32.15781
## disp      -0.04166  0.004944 -0.05134 -0.03175
```

### Posterior Predictive Distribution


```r
D <- seq(min(mtcars$disp), max(mtcars$disp))

# Samples from posterior predictive distribution
ppd <- as.data.frame(predict(mdl1, newdata=data.frame(disp=D)))
# Samples from expected value of posterior predictive distribution
eppd <- posterior_epred(mdl1, newdata=data.frame(disp=D), summary=FALSE) %>%
  head(50) %>%
  t() %>%
  as.data.frame() %>%
  mutate(disp=D) %>%
  pivot_longer(-disp, names_to="iter", values_to="mpg")


ggplot() +
  geom_ribbon(data=ppd, mapping=aes(x=D, ymin=Q2.5, ymax=Q97.5), alpha=0.5, fill="lightblue") +
  geom_line(data=eppd, mapping=aes(x=disp, y=mpg, group=iter), alpha=0.2) +
  geom_point(data=mtcars, mapping=aes(x=disp, y=mpg))
```

<img src="04_brms_files/figure-html/mdl1_ppd-1.png" width="672" />

## Semi-parametric Model

### Define Model

First, I'll define the splines just as I did with the `rethinking` package.  


```r
library(splines)

num_knots <- 4  # number of interior knots
knot_list <- quantile(mtcars$disp, probs=seq(0,1,length.out = num_knots))
B <- bs(mtcars$disp, knots=knot_list[-c(1,num_knots)], intercept=TRUE)

df1 <- cbind(disp=mtcars$disp, B) %>%
  as.data.frame() %>%
  pivot_longer(-disp, names_to="spline", values_to="val")

# Plot at smaller intervals so curves are smooth
N<- 50
D <- seq(min(mtcars$disp), max(mtcars$disp), length.out = N)
B_plot <- bs(D, 
             knots=knot_list[-c(1,num_knots)], 
             intercept=TRUE)

df2 <- cbind(disp=D, B_plot) %>%
  as.data.frame() %>%
  pivot_longer(-disp, names_to="spline", values_to="val")

ggplot(mapping=aes(x=disp, y=val, color=spline)) +
  geom_point(data=df1) +
  geom_line(data=df2, linetype="dashed")
```

<img src="04_brms_files/figure-html/splines-1.png" width="672" />

Note: the dashed lines are the splines and the points are the values of the spline at the specific values of `mtcars$disp`; the points are inputs into the `stan` model.



### Prior Predictive Distribution



### Diagnostics



### Posterior Distribution



### Posterior Predictive Distribution



## Semi-parametric Model (Random Walk Prior)

### Define Model

### Prior Predictive Distribution

### Diagnostics

### Posterior Distribution

### Posterior Predictive Distribution


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
##  [1] bayesplot_1.7.2 brms_2.14.4     Rcpp_1.0.5      forcats_0.5.0  
##  [5] stringr_1.4.0   dplyr_1.0.2     purrr_0.3.4     readr_1.4.0    
##  [9] tidyr_1.1.2     tibble_3.0.4    ggplot2_3.3.2   tidyverse_1.3.0
## 
## loaded via a namespace (and not attached):
##   [1] minqa_1.2.4          colorspace_2.0-0     ellipsis_0.3.1      
##   [4] ggridges_0.5.2       rsconnect_0.8.16     markdown_1.1        
##   [7] base64enc_0.1-3      fs_1.5.0             rstudioapi_0.13     
##  [10] farver_2.0.3         rstan_2.21.2         DT_0.16             
##  [13] fansi_0.4.1          mvtnorm_1.1-1        lubridate_1.7.9.2   
##  [16] xml2_1.3.2           codetools_0.2-16     bridgesampling_1.0-0
##  [19] knitr_1.30           shinythemes_1.1.2    projpred_2.0.2      
##  [22] jsonlite_1.7.1       nloptr_1.2.2.2       broom_0.7.2         
##  [25] dbplyr_2.0.0         shiny_1.5.0          compiler_4.0.3      
##  [28] httr_1.4.2           backports_1.2.0      assertthat_0.2.1    
##  [31] Matrix_1.2-18        fastmap_1.0.1        cli_2.2.0           
##  [34] later_1.1.0.1        prettyunits_1.1.1    htmltools_0.5.0     
##  [37] tools_4.0.3          igraph_1.2.6         coda_0.19-4         
##  [40] gtable_0.3.0         glue_1.4.2           reshape2_1.4.4      
##  [43] V8_3.4.0             cellranger_1.1.0     vctrs_0.3.5         
##  [46] nlme_3.1-149         crosstalk_1.1.0.1    xfun_0.19           
##  [49] ps_1.4.0             lme4_1.1-26          rvest_0.3.6         
##  [52] mime_0.9             miniUI_0.1.1.1       lifecycle_0.2.0     
##  [55] renv_0.12.0          gtools_3.8.2         statmod_1.4.35      
##  [58] MASS_7.3-53          zoo_1.8-8            scales_1.1.1        
##  [61] colourpicker_1.1.0   hms_0.5.3            promises_1.1.1      
##  [64] Brobdingnag_1.2-6    parallel_4.0.3       inline_0.3.17       
##  [67] shinystan_2.5.0      curl_4.3             gamm4_0.2-6         
##  [70] yaml_2.2.1           gridExtra_2.3        StanHeaders_2.21.0-6
##  [73] loo_2.3.1            stringi_1.5.3        dygraphs_1.1.1.6    
##  [76] pkgbuild_1.1.0       boot_1.3-25          rlang_0.4.9         
##  [79] pkgconfig_2.0.3      matrixStats_0.57.0   evaluate_0.14       
##  [82] lattice_0.20-41      labeling_0.4.2       rstantools_2.1.1    
##  [85] htmlwidgets_1.5.2    processx_3.4.5       tidyselect_1.1.0    
##  [88] plyr_1.8.6           magrittr_2.0.1       bookdown_0.21       
##  [91] R6_2.5.0             generics_0.1.0       DBI_1.1.0           
##  [94] pillar_1.4.7         haven_2.3.1          withr_2.3.0         
##  [97] mgcv_1.8-33          xts_0.12.1           abind_1.4-5         
## [100] modelr_0.1.8         crayon_1.3.4         rmarkdown_2.5       
## [103] grid_4.0.3           readxl_1.3.1         callr_3.5.1         
## [106] threejs_0.3.3        reprex_0.3.0         digest_0.6.27       
## [109] xtable_1.8-4         httpuv_1.5.4         RcppParallel_5.0.2  
## [112] stats4_4.0.3         munsell_0.5.0        shinyjs_2.0.0
```

