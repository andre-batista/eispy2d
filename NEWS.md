# Commit 01/11/2022
* Fixed bugs in `EvolutionaryAlgorithm` regarding printing information.
* Fixed bugs regarding axis size in multiple figures.
* When calling `InputData.draw`, the title of the figure should be given through `title` argument.
* Removed `path` attribute from `InputData`.
* Fixed bugs regarding Landweber and Conjugated-Gradient regularizers.
* Added Randomized Complete Block Design (RCBD) and Friedman's test as routines for multiple comparison among algorithms in the same test set.
* Added option to import algorithm configuration from constructor.
* Added Singular Value Decomposition (SVD) to reguralizers.
* Objects of `Regularization` can solve multiple right-hand-side inputs.
* Added analytical forward solution of the circular cylinder for PEC or perfect dielectric cases.

# Commit 10/25/2021
* Added notebook explaining the design of specific metrics for location and shape retrievement.

# Commit 10/21/2021
* Fixed use of `tight_layout` Matplolib's function when plotting multiple figures and when plotting the total field stored in the `InputData` class.
* Added the argument `mode` to `CaseStudy.reconstruction` to allow display only the best or all reconstructions done by the stochastic algorithms executions.

# Commit 10/20/2021
* Initial commit
