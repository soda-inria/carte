"""The setup script."""

from setuptools import setup, find_packages

with open('History.rst') as history_file:
    history = history_file.read()

requirements = []
test_requirements = []

setup(
    author="""Myung Jun Kim, Léo Grinsztajn, Gaël Varoquaux""",
    author_email='test@gmail.com',
    python_requires='>=3.10.12',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Environment :: Console',
        'Operating System :: OS Independent',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS',
        'Operating System :: POSIX',
        'Operating System :: Microsoft :: Windows',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    description="""Pretrained deep-learning models are the go-to solution for images or text. However, for tabular data the standard is still to train tree-based models.
      Indeed, transfer learning on tables hits the challenge of data integration: finding correspondences,
       correspondences in the entries (entity matching) where different words may denote the same entity, correspondences across columns (schema matching),
        which may come in different orders, names... We propose a neural architecture that does not need such correspondences.
         As a result, we can pretrain it on background data that has not been matched. 
         The architecture -- CARTE for Context Aware Representation of Table Entries -- uses a graph representation of tabular (or relational) data to process tables with different columns,
        string embedding of entries and columns names to model an open vocabulary, and a graph-attentional network to contextualize entries with column names and neighboring entries.
         An extensive benchmark shows that CARTE facilitates learning, outperforming a solid set of baselines including the best tree-based models.
           CARTE also enables joint learning across tables with unmatched columns, enhancing a small table with bigger ones. CARTE opens the door to large pretrained models for tabular data.""",
    install_requires=["numpy", "pandas", "scipy", "scikit-learn", "skrub","torch","torch-geometric","torcheval","torch_scatter"],
    license="MIT license",
    keywords='carte',
    name='carte',
    packages=find_packages(include=['carte', 'carte.*']),
    include_package_data=True,
    #package_data = {
    #    '': ['*.csv'],
    #    'carte': ['data/data_singletable/',"data/etc"],
    #},
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/soda-inria/carte',
    version='0.0.9',
    zip_safe=False,
)