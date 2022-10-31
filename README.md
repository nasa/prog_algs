# Prognostics Algorithm Python Package
[![CodeFactor](https://www.codefactor.io/repository/github/nasa/prog_algs/badge)](https://www.codefactor.io/repository/github/nasa/prog_algs)
[![GitHub License](https://img.shields.io/badge/License-NOSA-green)](https://github.com/nasa/prog_algs/blob/master/license.pdf)
[![GitHub Releases](https://img.shields.io/github/release/nasa/prog_algs.svg)](https://github.com/nasa/prog_algs/releases)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/nasa/prog_algs/master?tutorial.ipynb)

The Prognostic Algorithm Package is a python framework for model-based prognostics (computation of remaining useful life) of engineering systems, and provides a set of algorithms for state estimation and prediction, including uncertainty propagation. The algorithms take as inputs prognostic models (from NASA's Prognostics Model Package), and perform estimation and prediction functions. The library allows the rapid development of prognostics solutions for given models of components and systems. Different algorithms can be easily swapped to do comparative studies and evaluations of different algorithms to select the best for the application at hand.

This is designed to be used with the [Prognostics Models Package](https://github.com/nasa/prog_models).

## Installation
`pip3 install prog_algs`

## Documentation
See documentation [here](https://nasa.github.io/progpy/prog_algs_guide.html)

## Repository Directory Structure 

`src/prog_algs/` - The prognostics algorithm python package<br />
`docs/` - Project documentation (see also [github.io](https://nasa.github.io/prog_algs/))<br />
`examples/` - Example Python scripts using prog_algs<br />
`tests/` - Tests for prog_models<br />
`README.md` - The readme (this file)<br />

## Citing this repository
Use the following to cite this repository:

```
@misc{2022_nasa_prog_algs,
    author    = {Christopher Teubert and Matteo Corbetta and Chetan Kulkarni},
    title     = {Prognostics Algorithm Python Package},
    month     = Oct,
    year      = 2022,
    version   = {1.4.0},
    url       = {https://github.com/nasa/prog\_algs}
    }
```

The corresponding reference should look like this:

C. Teubert, M. Corbetta, C. Kulkarni, Prognostics Algorithm Python Package, v1.4.0, Oct 2022. URL https://github.com/nasa/prog_algs.

Alternatively, if using both prog_models and prog_algs, you can cite the combined package as

C. Teubert, C. Kulkarni, M. Corbetta, K. Jarvis, M. Daigle, ProgPy Prognostics Python Packages, v1.4, October 2022. URL https://nasa.github.io/progpy.

## Acknowledgements
The structure and algorithms of this package are strongly inspired by the [MATLAB Prognostics Algorithm Library](https://github.com/nasa/PrognosticsAlgorithmLibrary) and the [MATLAB Prognostics Metrics Library](https://github.com/nasa/PrognosticsMetricsLibrary). We would like to recognize Matthew Daigle, Shankar Sankararaman and the rest of the team that contributed to the Prognostics Model Library for the contributions their work on the MATLAB library made to the design of prog_algs

## Notices
Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

## Disclaimers
No Warranty: THE SUBJECT SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY WARRANTY OF ANY KIND, EITHER EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR FREEDOM FROM INFRINGEMENT, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL BE ERROR FREE, OR ANY WARRANTY THAT DOCUMENTATION, IF PROVIDED, WILL CONFORM TO THE SUBJECT SOFTWARE. THIS AGREEMENT DOES NOT, IN ANY MANNER, CONSTITUTE AN ENDORSEMENT BY GOVERNMENT AGENCY OR ANY PRIOR RECIPIENT OF ANY RESULTS, RESULTING DESIGNS, HARDWARE, SOFTWARE PRODUCTS OR ANY OTHER APPLICATIONS RESULTING FROM USE OF THE SUBJECT SOFTWARE.  FURTHER, GOVERNMENT AGENCY DISCLAIMS ALL WARRANTIES AND LIABILITIES REGARDING THIRD-PARTY SOFTWARE, IF PRESENT IN THE ORIGINAL SOFTWARE, AND DISTRIBUTES IT "AS IS."

Waiver and Indemnity:  RECIPIENT AGREES TO WAIVE ANY AND ALL CLAIMS AGAINST THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR RECIPIENT.  IF RECIPIENT'S USE OF THE SUBJECT SOFTWARE RESULTS IN ANY LIABILITIES, DEMANDS, DAMAGES, EXPENSES OR LOSSES ARISING FROM SUCH USE, INCLUDING ANY DAMAGES FROM PRODUCTS BASED ON, OR RESULTING FROM, RECIPIENT'S USE OF THE SUBJECT SOFTWARE, RECIPIENT SHALL INDEMNIFY AND HOLD HARMLESS THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR RECIPIENT, TO THE EXTENT PERMITTED BY LAW.  RECIPIENT'S SOLE REMEDY FOR ANY SUCH MATTER SHALL BE THE IMMEDIATE, UNILATERAL TERMINATION OF THIS AGREEMENT.
