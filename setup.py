"""
setup.py
"""
import os
from codecs import open
from setuptools import setup, find_packages
from setuptools.command.develop import develop
from subprocess import check_call
import shlex
from warnings import warn


class PostDevelopCommand(develop):
    """
    Class to run post setup commands
    """

    def run(self):
        """
        Run method that tries to install pre-commit hooks
        """
        try:
            check_call(shlex.split("pre-commit install"))
        except Exception as e:
            warn("Unable to run 'pre-commit install': {}"
                 .format(e))

        develop.run(self)


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

with open("requirements.txt") as f:
    install_requires = f.readlines()

test_requires = ["pytest>=5.2", ]
description = "National Solar Radiation DataBase"

setup(
    name="NREL-nsrdb",
    version=version,
    description=description,
    long_description=readme,
    author="Grant Buster",
    author_email="grant.buster@nrel.gov",
    url="https://github.com/NREL/",
    packages=find_packages(),
    package_dir={"nsrdb": "nsrdb"},
    include_package_data=True,
    zip_safe=False,
    keywords="nsrdb",
    python_requires='>=3.6',
    classifiers=[
        "Development Status :: Beta",
        "Intended Audience :: Modelers",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    test_suite="tests",
    install_requires=install_requires,
    extras_require={
        "test": test_requires,
        "dev": test_requires + ["flake8", "pre-commit", "pylint"],
    },
    cmdclass={"develop": PostDevelopCommand},
)
