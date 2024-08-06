from setuptools import setup, find_packages

setup(
    name="skin-project",
    version="0.0.1",
    author="Olusegun Ekundayo Adebayo",
    author_email="olusegun.adebayo@unvi-fcomte.fr",
    maintainer="Olusegun Ekundayo Adebayo",
    maintainer_email="olusegun.adebayo@unvi-fcomte.fr",
    description="To classify various skin disorders",
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    python_requires=">=3.9",
)