# Introduction {#intro}

## Resources

- [Bayesian Data Analysis](http://www.stat.columbia.edu/~gelman/book/) by Andrew Gelman, et. al.

- [A First Course in Bayesian Statistical Methods](https://pdhoff.github.io/book/) by Peter Hoff

- [Statistical Rethinking](https://xcelab.net/rm/statistical-rethinking/) by Richard McElreath

## A Motivating Example

Let's start with something familiar--the Monty Hall problem.  There are three doors labeled A, B and C. A car is behind one of the doors and a goat is behind each of the other two doors. You choose a door (let's say A). Monty Hall, who knows where the car actually is, opens one of the other doors (let's say B) revealing a goat.  Do you stay with door A or do you switch to door C?

We can frame the problem as follows:

1. Initially, you believe the car is equally likely to behind each door (i.e., $P[A=car]=P[B=car]=P[C=car]=\frac{1}{3}$).  Let's call this the _prior_ information.

2. Next, you can calculate the conditional probabilities that Monty Hall opened door B. Let's call this the _likelihood_.

\begin{align*}
  P[B=open | A=car] &= \frac{1}{2}\\
  P[B=open | B=car] &= 0 \\
  P[B=open | C=car] &=1
\end{align*}

3. Finally, you can update your beliefs with the new information.  Let's call your updated beliefs the _posterior_.

\begin{align*}
  P[A=car|B=open] &= \frac{P[B=open|A=car]P[A=car]}{P[B=open]} = \frac{1/2 * 1/3}{1/6 + 0 + 1/3} = \frac{1}{3} \\
  P[B=car|B=open] &= \frac{P[B=open|B=car]P[B=car]}{P[B=open]} = \frac{0 * 1/3}{1/6 + 0 + 1/3} = 0 \\
  P[C=car|B=open] &= \frac{P[B=open|C=car]P[C=car]}{P[B=open]} = \frac{1 * 1/3}{1/6 + 0 + 1/3} = \frac{2}{3}
\end{align*}
  
Clearly, you should switch to door C.  

This is a toy illustration of how to think about a model in a Bayesian framework:

$$posterior \propto likelihood * prior$$
(See the resources for a proper mathematical derivation.)

## Workflow

The workflow I'll follow in the subsequent chapters is as follows:

1. Define the model.
2. Examine the prior predictive distribution.
3. Examine diagnostic plots.
4. Examine posterior distribution.
5. Examine the posterior predictive distribution.

In general, this is an iterative process. At each step you may discover something that causes you to start over at step 1 with a new, refined model. 

## Data

For all of the examples, I use the mtcars data set and a model with _disp_ as the predictor and _mpg_ as the response. I start with a simple linear regression. However, as you can see from the scatterplot below, the relationship between _mpg_ and _disp_ is not linear, so I also fit a slightly more complex semi-parametric model. 


```r
library(tidyverse)
library(datasets)
data(mtcars)
mtcars %>%
  ggplot(aes(x=disp, y=mpg)) +
  geom_point() 
```

<img src="01_intro_files/figure-html/mtcars-1.png" width="672" />
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
## [1] stats     graphics  grDevices datasets  utils     methods   base     
## 
## other attached packages:
## [1] forcats_0.5.0   stringr_1.4.0   dplyr_1.0.2     purrr_0.3.4    
## [5] readr_1.4.0     tidyr_1.1.2     tibble_3.0.4    ggplot2_3.3.2  
## [9] tidyverse_1.3.0
## 
## loaded via a namespace (and not attached):
##  [1] tidyselect_1.1.0  xfun_0.19         haven_2.3.1       colorspace_2.0-0 
##  [5] vctrs_0.3.5       generics_0.1.0    htmltools_0.5.0   yaml_2.2.1       
##  [9] rlang_0.4.9       pillar_1.4.7      glue_1.4.2        withr_2.3.0      
## [13] DBI_1.1.0         dbplyr_2.0.0      modelr_0.1.8      readxl_1.3.1     
## [17] lifecycle_0.2.0   munsell_0.5.0     gtable_0.3.0      cellranger_1.1.0 
## [21] rvest_0.3.6       evaluate_0.14     labeling_0.4.2    knitr_1.30       
## [25] fansi_0.4.1       broom_0.7.2       Rcpp_1.0.5        renv_0.12.0      
## [29] scales_1.1.1      backports_1.2.0   jsonlite_1.7.1    farver_2.0.3     
## [33] fs_1.5.0          hms_0.5.3         digest_0.6.27     stringi_1.5.3    
## [37] bookdown_0.21     grid_4.0.3        cli_2.2.0         tools_4.0.3      
## [41] magrittr_2.0.1    crayon_1.3.4      pkgconfig_2.0.3   ellipsis_0.3.1   
## [45] xml2_1.3.2        reprex_0.3.0      lubridate_1.7.9.2 assertthat_0.2.1 
## [49] rmarkdown_2.5     httr_1.4.2        rstudioapi_0.13   R6_2.5.0         
## [53] compiler_4.0.3
```



