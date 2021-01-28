"""
setup.py
"""
import os
from codecs import open
from setuptools import setup, find_packages
import sys

py_version = sys.version_info
if py_version.major < 3:
    raise RuntimeError("NSRDB is not compatible with python 2!")
elif py_version.minor <= 6:
    raise RuntimeError("NSRDB is not compatible with python <= 3.6!")

try:
    from pypandoc import convert_text
except ImportError:
    convert_text = lambda string, *args, **kwargs: string

here = os.path.abspath(os.path.dirname(__file__))

with open("README.md", encoding="utf-8") as readme_file:
    readme = convert_text(readme_file.read(), "md", format="md")

with open(os.path.join(here, "nsrdb", "version.py"), encoding="utf-8") as f:
    version = f.read()

version = version.split('=')[-1].strip().strip('"').strip("'")

setup(
    name="nsrdb",
    version=version,
    description="The National Solar Radiation DataBase",
    long_description=readme,
    author="Grant Buster",
    author_email="grant.buster@nrel.gov",
    url="https://github.nrel.gov/PXS/nsrdb",
    packages=find_packages(),
    package_dir={"nsrdb": "nsrdb"},
    entry_points={
        "console_scripts": [
            "nsrdb=nsrdb.cli:main",
        ],
    },
    include_package_data=True,
    license="BSD license",
    zip_safe=False,
    keywords="nsrdb",
    classifiers=[
        "Development Status :: Beta",
        "Intended Audience :: Modelers",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    test_suite="tests",
    install_requires=["numpy>=1.16",
                      "pandas>=0.25,<1",
                      "click>=7.0",
                      "scipy>=1.3",
                      "pyhdf",
                      "h5py>=3.1.0",
                      "scikit-learn>=0.21",
                      "netcdf4>=1.4",
                      "matplotlib>=3.1",
                      "pytest>=5.2",
                      "ipython",
                      "notebook",
                      "psutil",
                      "pre-commit",
                      "flake8",
                      "pylint",
                      "NREL-rex>=0.2.30",
                      "NREL-reV>=0.4.37",
                      ],
)
