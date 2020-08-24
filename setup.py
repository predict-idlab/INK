import setuptools

with open("readme.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="semantic_ink",
    version="0.0.1",
    author="Bram Steenwinckel",
    author_email="bram.steenwinckel@ugent.be",
    description="INK: Instance Neighbouring by using Knowledge",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="Knowledge graph representation, Semantic Rule Mining",
    url="https://github.com/IBCNServices/INK",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: IMEC License",
        "Operating System :: OS Independent",
    ],
    project_urls={
        "Source": "https://github.com/IBCNServices/INK",
        "Tracker": "https://github.com/IBCNServices/INK/issues",
    },
)