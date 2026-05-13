# eispy2d

An Open-Source Python Library for the development of algorithms for 2D Electromagnetic Inverse Scattering Problems (EISPs).

## Motivation

This library was thought to provide a common and basic framework for researchers that want to test new ideas about algorithms for EISPs. Then, they will not need to develop the whole structure (domain model, discretization formulations, forward solvers, data visualization, statistical inference, etc).

## What can I do with this library?

With the tools in this library, you can represent an instance of EISP, develop algorithms, run them, and analyze the results in many different ways. The library provides specific implementations for case studies and benchmarking, so one can get preliminary results, measure the performance, and compare with different algorithms or different versions of the same algorithm.

## Model assumptions

Besides considering the two-dimensional formulation, we are assuming as well TMz polarization of incident waves and linear, isotropic, non-dispersive, and non-magnetic materials.

## Install

Initially, the library was thought to be a collection of ".py" files that anyone can download and add to his/her project. It would be amazing if, one day, this library became a well-organized Python package which one can install through Pip or Conda. But, as this is an implementation developed by only one person, then these steps will be considered someday in the future. For while, you just need to [download the codes](https://github.com/andre-batista/eispy2d/tree/main/lib) and call the modules as you do with any library that you create.

## Dependencies

Before using eispy2d, please ensure you have all the required dependencies installed. The project includes a `requirements.txt` file listing all necessary packages. You can install these dependencies by running `pip install -r requirements.txt` in your command line or terminal. These packages are essential for the proper functioning of the library, including numerical computations, visualization tools, and optimization algorithms that power the electromagnetic inverse scattering solvers.

## How to use

You may find usages examples [here](https://github.com/andre-batista/eispy2d/tree/main/demo). There are scripts and Jupyter Notebooks in which you can see how the classes are called, how to build a problem, how to run an experiment, etc.

## Documentation

Documentation is currently being built and a previous version can be found in https://eispy2d-docs.readthedocs.io/en/latest/api.html

## Contribute

**You are totally welcome to contribute to this library** by finding bugs, suggesting changes, implementing the algorithms in the literature, and providing your algorithms so others can use them to compare in their experiments. You may add issues, send pull requests or contact me through e-mail.

## Citation

We've already written an article describing the library. While it is still under review, its *preprint* version is available at the arXiv repository via this [link](https://arxiv.org/abs/2111.02185#). If you use this library, you may acknowledge by citing it:

```
@ARTICLE{11015426,
  author={Costa Batista, André and Adriano, Ricardo and Batista, Lucas S.},
  journal={IEEE Access}, 
  title={EISPY2D: An Open-Source Python Library for the Development and Comparison of Algorithms in Two-Dimensional Electromagnetic Inverse Scattering Problems}, 
  year={2025},
  volume={13},
  number={},
  pages={92134-92154},
  keywords={Libraries;Electromagnetic scattering;Imaging;Image reconstruction;Electromagnetics;Microwave theory and techniques;Microwave integrated circuits;Microwave imaging;Microwave FET integrated circuits;Inverse problems;Comparison of algorithms;electromagnetic inverse scattering problem;microwave imaging;open-source library;optimization},
  doi={10.1109/ACCESS.2025.3573679}
}
```

## Further information

For further information and questions, please send me an [email](andre-costa@ufmg.br).

Have fun!
André
