--- 
title: "Intro to R Bayes Packages"
author: "Melissa Wong"
date: "2021-01-02"
site: bookdown::bookdown_site
output: bookdown::gitbook
documentclass: book
bibliography: [packages.bib, book.bib]
nocite: '@*'
biblio-style: apalike
link-citations: yes
github-repo: melissa-wong/r-ladies-rtp-bayes-pkgs
description: "This is an introduction to several R packages for Bayesian analysis."
---



# Preface {-}

My motivation for this presentation was to put together the "Intro to R packages for Bayesian models" information I wish had gotten in the Intro to Bayes course I took in grad school.  During that course, we learned some of the underlying theory, spent a _lot_ of time on conjugate priors and deriving posterior distributions by hand, and implemented some sampling algorithms. All great things to learn!  However, after the class was over I didn't feel like I could actually use what I learned in practice. It just seemed silly to me to think I would write a custom sampler every time when there are existing software packages that use better, faster sampling algorithms than what we covered in class. So then the question became which software package should I use and why? This presentation summarizes what I learned by experimenting with several different packages in the process of answering that question for myself. Hopefully the information that follows will also help you figure out which path is easiest for you to start using Bayesian methods.


