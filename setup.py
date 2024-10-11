from setuptools import setup, find_packages

# Read the long description from README.md
with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

requirements = [
    "numpy", "pandas", "scipy", "scikit-learn", "skrub",
    "torch", "torch-geometric", "torcheval","catboost","fasttext","category-encoders","tabpfn","xgboost",
]

setup(
    author="Myung Jun Kim, Léo Grinsztajn, Gaël Varoquaux",
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
    description="CARTE-AI: Context Aware Representation of Table Entries for AI",
    long_description=long_description,
    long_description_content_type='text/markdown',  # Adjust if using RST
    install_requires=requirements,
    license="MIT license",
    keywords='carte-ai',
    name='carte-ai',
    packages=find_packages(include=['carte_ai', 'carte_ai.*']),
    include_package_data=True,
    package_data={
        'carte_ai': [
            'data/spotify.parquet',
            'data/wine_pl.parquet',
            'data/wine_dot_com_prices.parquet',
            'data/wine_vivino_price.parquet',
            'data/spotify/config_.json',
            'data/config_wine_pl.json',
            'data/config_wine_dot_com_prices.json',
            'data/config_wine_vivino_price.json',
            'data/etc/kg_pretrained.pt',
        ],
    },
    extras_require={
        'test': ['pytest', 'coverage'],  # Add your test requirements here
    },
    url='https://github.com/soda-inria/carte',
    version='0.0.8',
    zip_safe=False,
)
