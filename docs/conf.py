# Configuration file for the Sphinx documentation builder.

# -- Path setup --------------------------------------------------------------
import os
import sys
dirname = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(dirname, ".."))

# -- Project information -----------------------------------------------------

project = 'nnodely'
author = 'tonegas'
release = '1.0'
version = '1.0'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

templates_path = []
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = []

# -- Options for EPUB output -------------------------------------------------
epub_copyright = '2024, tonegas'  # Add this line