'''
Authors : Gastone Pietro Rosati Papini, Sebastiano Taddei
Date    : 31/10/2022
License : MIT
'''

from setuptools import find_packages, setup

from neu4mes import __version__

with open('README.md', 'r', encoding='utf-8') as fh:
    readme_file = fh.read()

with open('LICENSE', 'r', encoding='utf-8') as fh:
    license_file = fh.read()

setup(
    name='neu4mes',
    version=__version__,
    author='Gastone Pietro Rosati Papini, Sebastiano Taddei',
    author_email='tonegas@gmail.com, sebastianotaddei@gmail.com',
    description=('Mechanics-informed neural network framework ',
                 'for modeling and control mechanical system'),
    long_description=readme_file,
    long_description_content_type='text/markdown',
    url='https://github.com/tonegas/neu4mes',
    license=license_file,
    packages=find_packages(exclude=('docs', 'examples', 'tests')),
    platforms='any',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    install_requires=['numpy','pandas','tensorflow','pyyaml'],
    python_requires='>3.10.6'
)
