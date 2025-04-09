"""STAR SHINE

Code written by: Luc IJspeert
"""

from setuptools import setup


# package version
MAJOR = 0
MINOR = 1
ATTR = '0'
# full acronym
ACRONYM = ("Satellite Time-series Analysis Routine using "
           "Sinusoids and Harmonics through Iterative Non-linear Extraction")

setup(name="star_shine",
      version=f"{MAJOR}.{MINOR}.{ATTR}",
      author="Luc IJspeert",
      url="https://github.com/LucIJspeert/star_shine",
      license="GNU General Public License v3.0",
      description=ACRONYM,
      long_description=open('README.md').read(),
      packages=['star_shine'],
      package_dir={'star_shine': 'star_shine'},
      package_data={'star_shine': ['data/*']},
      include_package_data=True,
      python_requires=">=3.6",
      install_requires=['numpy', 'scipy', 'numba', 'h5py', 'astropy', 'pandas', 'matplotlib', 'yaml'],
      extras_require={'mcmc': ['pymc3', 'fastprogress', 'theano', 'arviz']}
     )
