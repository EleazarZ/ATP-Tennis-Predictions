############################################################################
# setup.py file describes the project and the files that belong to it
# Project Layout
###########################################################################

from setuptools import find_packages, setup

setup(
    name="atp",
    version="1.0.0",
    packages=find_packages(),
    include_packages_data=True,
    zip_safe=False,
    install_requires=["flask", "dash"],
    description="ATP tennis match pronostic",
    author="e-z",
    author_email="firstname.name@domain.fr",
    entry_point={},
)
