# eispy2d

An Open-Source Python Library for the development of algorithms for 2D Electromagnetic Inverse Scattering Problems (EISPs).

## Motivation

This library was thought to provide a common and basic framework for researchers that want to test new ideas about algorithms for EISPs. Then, they will not need to develop the whole structure (domain model, discretization formulations, forward solvers, data visualization, statistical inference, etc).

## What can I do with this library?

With the tools in this library, you can represent an instance of EISP, develop algorithms, run them, and analyze the results in many different ways. The library provides specific implementations for case studies and benchmarking, so one can get preliminary results, measure the performance, and compare with different algorithms or different versions of the same algorithm.

## Model assumptions

Besides considering the two-dimensional formulation, we are assuming as well TMz polarization of incident waves and linear, isotropic, non-dispersive, and non-magnetic materials.

## Install

Initially, the library was thought to be a collection of ".py" files that anyone can download and add to his/her project. It would be amazing if, one day, this library became a well-organized Python package which one can install through Pip or Conda. But, as this is an implementation developed by only one person who is pursuing his Ph.D. degree, then these steps will be considered someday in the future. For while, you just need to [download the codes](https://github.com/andre-batista/eispy2d/tree/main/lib) and call the modules as you do with any library that you create.

But, pay attention: **there are packages that you must install in order to run the codes!** These are packages that you can install through Pip or Conda. So, that should be an easy thing. Here is the list of the required packages:

* Numpy
* Scipy
* Matplotlib
* Numba
* Pickle
* Statsmodels
* Joblib
* Multiprocessing
* Skimage
* Pingouin

## How to use

You may find usages examples [here](https://andre-batista.github.io/eispy2d/usage-examples.html). There are scripts and Jupyter Notebooks in which you can see how the classes are called, how to build a problem, how to run an experiment, etc.

## Contribute

**You are totally welcome to contribute to this library** by finding bugs, suggesting changes, implementing the algorithms in the literature, and providing your algorithms so others can use them to compare in their experiments. You may add issues, send pull requests or contact me through e-mail.

## Citation

The paper is under review. So, we hope that soon you will be able to acknowledge by citing it.

## Further information

For further information and questions, please send me an [email](andre-costa@ufmg.br).

Have fun!
André
