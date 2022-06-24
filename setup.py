from setuptools import setup, find_namespace_packages

REQUIREMENTS = open("requirements.txt").read().splitlines()

setup(
    name="amazing_ai",
    description="Random stuff for my Bachelor thesis",
    author="Lennart Kämmle",
    author_email="lennart.kaemmle@desy.de",
    packages=find_namespace_packages(include=("amazing_ai",)),
    install_requires=REQUIREMENTS
)
