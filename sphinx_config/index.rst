Prognostics Algorithms Python Package
=============================================================
.. image:: https://mybinder.org/badge_logo.svg
 :target: https://mybinder.org/v2/gh/nasa/prog_algs/master?labpath=tutorial.ipynb

The NASA PCoE Prognostic Algorithms Package is a python framework for model-based prognostics (computation of remaining useful life) of engineering systems. The package provides an extendable set of algorithms for state estimation and prediction, including uncertainty propagation. The package also include metrics, visualization, and analysis tools needed to measure the prognostic performance. The algorithms use prognostic models (from NASA's Prognostics Model Package) to perform estimation and prediction functions. The package enables the rapid development of prognostics solutions for given models of components and systems. Different algorithms can be easily swapped to do comparative studies and evaluations of different algorithms to select the best for the application at hand.

The Prognostics Algorithms Package was developed by researchers of the NASA Prognostics Center of Excellence (PCoE) and `Diagnostics & Prognostics Group <https://www.nasa.gov/content/diagnostics-prognostics>`__.

If you are new to this package, see `getting started <getting_started.html>`__.

.. toctree::
   :maxdepth: 2
   
   Tutorial <https://mybinder.org/v2/gh/nasa/prog_algs/master?labpath=tutorial.ipynb>
   getting_started
   state_estimators
   predictors
   uncertain_data
   prediction
   metrics
   ProgModels <https://nasa.github.io/prog_models>
   ProgServer <https://nasa.github.io/prog_server>
   dev_guide <https://nasa.github.io/prog_models/dev_guide.html>
   GitHub <https://github.com/nasa/prog_algs>
   release

Citing this repository
-----------------------
Use the following to cite this repository:

@misc{2021_nasa_prog_algs,
  | author    = {Christopher Teubert and Chetan Kulkarni and Matteo Corbetta},
  | title     = {Prognostics Algorithms Python Package},
  | month     = May,
  | year      = 2022,
  | version   = {1.3.0},
  | url       = {https://github.com/nasa/prog_algs}
  | }

The corresponding reference should look like this:

C. Teubert, C. Kulkarni, M. Corbetta. Prognostics Algorithms Python Package, v1.3.0, May 2022. URL https://github.com/nasa/prog_algs.

Indices and tables
-----------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Disclaimers
----------------------

No Warranty: THE SUBJECT SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY WARRANTY OF ANY KIND, EITHER EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR FREEDOM FROM INFRINGEMENT, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL BE ERROR FREE, OR ANY WARRANTY THAT DOCUMENTATION, IF PROVIDED, WILL CONFORM TO THE SUBJECT SOFTWARE. THIS AGREEMENT DOES NOT, IN ANY MANNER, CONSTITUTE AN ENDORSEMENT BY GOVERNMENT AGENCY OR ANY PRIOR RECIPIENT OF ANY RESULTS, RESULTING DESIGNS, HARDWARE, SOFTWARE PRODUCTS OR ANY OTHER APPLICATIONS RESULTING FROM USE OF THE SUBJECT SOFTWARE.  FURTHER, GOVERNMENT AGENCY DISCLAIMS ALL WARRANTIES AND LIABILITIES REGARDING THIRD-PARTY SOFTWARE, IF PRESENT IN THE ORIGINAL SOFTWARE, AND DISTRIBUTES IT "AS IS."

Waiver and Indemnity:  RECIPIENT AGREES TO WAIVE ANY AND ALL CLAIMS AGAINST THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR RECIPIENT.  IF RECIPIENT'S USE OF THE SUBJECT SOFTWARE RESULTS IN ANY LIABILITIES, DEMANDS, DAMAGES, EXPENSES OR LOSSES ARISING FROM SUCH USE, INCLUDING ANY DAMAGES FROM PRODUCTS BASED ON, OR RESULTING FROM, RECIPIENT'S USE OF THE SUBJECT SOFTWARE, RECIPIENT SHALL INDEMNIFY AND HOLD HARMLESS THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR RECIPIENT, TO THE EXTENT PERMITTED BY LAW.  RECIPIENT'S SOLE REMEDY FOR ANY SUCH MATTER SHALL BE THE IMMEDIATE, UNILATERAL TERMINATION OF THIS AGREEMENT.