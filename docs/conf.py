"""Configuration file for the Sphinx documentation builder."""
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../'))

# Add type of source files
source_suffix = ['.rst', '.md']

master_doc = 'contents'

# -- Project information -----------------------------------------------------

project = 'PeriPy'
copyright = '2020, Jim Madge, Tim Dodwell, Ben Boys, Greg Mingas'
author = 'Jim Madge, Tim Dodwell, Ben Boys, Greg Mingas'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx'
]

# autodoc configuration
# Concatenate a class' docstring with the __init__ method docstring
autoclass_content = 'both'

# intersphinx configuration
intersphinx_mapping = {
    'numpy': ('https://docs.scipy.org/doc/numpy/', None),
    }

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'
html_theme_options = {
    'github_user': 'alan-turing-institute',
    'github_repo': 'probabilistic-peridynamics',
    'github_banner': True,
    'fixed_sidebar': True,
    'sidebar_width': '250px'
    }

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']
