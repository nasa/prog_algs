Developers Guide
================

This document includes some details relevant for developers. 

..  contents:: 
    :backlinks: top

Guidance for Contributors
-------------------------

Below are a few design decision made by the authors, documented for the reference of new contributors:

* When supplied by or to the user, values with names (e.g., inputs, states, outputs, event_states, event occurance, etc.) should be supplied as dictionaries (or dict-like objects) where they can be referred to by name. 
* Visualize and metrics subpackages shall be independent (i.e., not have any dependencies with the wider package or other subpackages)
* Limit introduction of new external dependencies, when possible. 
* This is a research tool, so when making a design decision between operational efficiency and usability, generally choose the more usable option
* Except in the most extreme cases, maintain backwards compatibility for the convenience of existing users

  * If a feature is to be removed, mark it as depreciated for at least 2 releases before removing

* Whenever possible, design so a new feature can be used for any model, interchangably. 
* Whenever possible, UncertainData types, state estimators, and predictors should be interchangable
* Demonstrate common use cases as an example. 
* Every feature should be demonstrated in an example

  * The most commonly used features should be demonstrated in the tutorial

Branching Strategy
------------------
Our project is following the git strategy described `here <https://nvie.com/posts/a-successful-git-branching-model/>`_. Details specific to each branch are described below. 

`master`: Every merge into the master branch is done using a pull request (never commiting directly), is assigned a release number, and must comply with the release checklist. The release checklist is a software assurance tool. 

`dev`: Every merge into the dev branch that contains a functional change is done using a pull request (not commiting directly). Every commit should be functional. All unit tests must function before commiting to dev or merging another branch. 

`Feature Branches`: These branches include changes specific to a new feature. Before merging into dev unit tests should all run, tests should be added for the feature, and documentation should be updated, as appropriate.

Release Checklist
*****************
* Code review - all software must be checked by someone other than the author
* Check that each new feature has a corresponding tests
* Run unit tests `python -m tests`
* Check documents- see if any updates are required
* Rebuild sphinx documents: `sphinx-build sphinx-config/ docs/`
* Write release notes
* For releases adding new features- ensure that NASA release process has been followed

NPR 7150
--------
This section describes this project's compliance with the NASA's NPR 7150 requirements, documented `here <https://nodis3.gsfc.nasa.gov/displayDir.cfm?t=NPR&c=7150&s=2B>`_.

* Software Classification: Class-E (Research Software)
* Safety Criticality: Not Safety Critical 

Compliance Notation Legend
**************************
* FC: Fully Compliant
* T: Tailored (Specific tailoring described in mitigation) `SWE-121 <https://swehb.nasa.gov/display/7150/SWE-121+-+Document+Alternate+Requirements>`_
* PC: Partially Compliant
* NC: Not Compliant
* NA: Not Applicable

Compliance Matrix
*****************
+-------+----------------------------------+------------+---------------------+
| SWE # | Description                      | Compliance | Evidence            |
+=======+==================================+============+=====================+
| 033   | Assess aquisiton Options         | FC         | See section below   |
+-------+----------------------------------+------------+---------------------+
| 013   | Maintain Software Plans          | FC         | This document       |
+-------+----------------------------------+------------+---------------------+
| 042   | Electronic Accesss to Source     | FC         | This repo           |
+-------+----------------------------------+------------+---------------------+
| 139   | Comply with 7150                 | FC         | This document       |
+-------+----------------------------------+------------+---------------------+
| 121   | Tailored Reqs                    | NA         | No tailoring        |
+-------+----------------------------------+------------+---------------------+
| 125   | Compliance Matrix                | FC         | This document       |
+-------+----------------------------------+------------+---------------------+
| 029   | Software Classification          | FC         | This document       |
+-------+----------------------------------+------------+---------------------+
| 022   | Software Assurance               | FC         | This document       |
+-------+----------------------------------+------------+---------------------+
| 205   | Safety Cricial Software          | FC         | See above           |
+-------+----------------------------------+------------+---------------------+
| 023   | Safety Critical Reqs             | NA         | Not safety critical |
+-------+----------------------------------+------------+---------------------+
| 206   | Autogen Software                 | NA         | No autogen          |
+-------+----------------------------------+------------+---------------------+
| 148   | Software Catolog                 | FC         | Will be added       |
+-------+----------------------------------+------------+---------------------+
| 156   | Perform CyberSecurity Assessment | FC         | See section below   |
+-------+----------------------------------+------------+---------------------+

Aquisition Options
******************
Assessed, there are some existing prognostics tools, but no general python package that can support model-based prognostics like we need. 

Cybersecurity Assessment 
************************
Assessed, no significant Cybersecurity concerns were identified- research software. 