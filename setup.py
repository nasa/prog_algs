from distutils.core import setup
setup(
    name = 'prog_algs',
    packages = ['prog_algs'],
    version = '1.0',
    license = 'NOSA',
    description = "The NASA Prognostic Algorithm Package is a python framework for model-based prognostics (computation of remaining useful life) of engineering systems, and provides a set of algorithms for state estimation and prediction, including uncertainty propagation. The algorithms take as inputs prognostic models (from NASA's Prognostics Model Package), and perform estimation and prediction functions. The library allows the rapid development of prognostics solutions for given models of components and systems. Different algorithms can be easily swapped to do comparative studies and evaluations of different algorithms to select the best for the application at hand.",
    author = 'Christopher Teubert',
    author_email = 'christopher.a.teubert@nasa.gov',
    url = '', # TBD
    download_url = '', # TBD
    keywords = ['prognostics', 'diagnostics', 'fault detection', 'prognostics and health management', 'PHM', 'health management'],
    install_requires = [
        'numpy',
        'scipy',
        'filtpy',
        'matplotlib'
    ],
    classifiers = [
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Researchers',     
        'Topic :: Research Tools :: Prognostics and Health Management',
        'License :: NASA Open Source Agreement',   
        'Programming Language :: Python :: 3',     
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7'
    ]
)