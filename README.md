# Presentation for R-Ladies RTP, 6 January 2021

## Environment Setup

This repository uses `renv` to manage packages.  If you don't already have `renv` install, simply type `install.packages(renv)`.  Then clone the repo, and type `renv::restore()` which will download all of the packages required to build the project.

## Build Instructions

After setting up the environment, there should be a "Build" tab in the Rstudio IDE next to the "Git" tab. (Note: If this tab is _not_ present, that indicates the problem was most likely with installing the `bookdown` package.). Click the "Build Book" button under the "Build" tab and that will build the entire book.
