import setuptools

from neu4mes import __version__

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="neu4mes",
    version=__version__,
    author="Gastone Pietro Rosati Papini",
    author_email="tonegas@gmail.com",
    description="Mechanics-informed neural network framework for modeling and control mechanical system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tonegas/neu4mes",
    packages=setuptools.find_packages(),
    platforms='any',
    packages=["neu4mes"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[ 'keras', 'tensorflow','numpy','PyYAML'],
    python_requires='>3.6'
)