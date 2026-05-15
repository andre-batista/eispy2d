# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


import os
import sys

sys.path.insert(0, os.path.abspath('../../')) 

project = 'eispy2d'
copyright = '2026, André Costa Batista'
author = 'André Costa Batista'
release = '1.0.3'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',      # for extracting docstrings
    'sphinx.ext.napoleon',     # for Google / NumPy style docstrings
    'sphinx_autodoc_typehints' # optional, for type hints
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_theme_options = {
    "github_url": "https://github.com/andre-batista/eispy2d.git",
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
}

html_static_path = ['_static']
html_title = 'Eispy2d Documentation'

html_show_sourcelink = False

# Better handling of docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True

latex_engine = 'xelatex'  
latex_use_xindy = False


latex_elements = {
    'inputenc': r'\usepackage[utf8]{inputenc}',
    'fontenc': r'\usepackage{fontspec}',  
    'preamble': r'''
        \usepackage{amsmath}
        \usepackage{amssymb}
        \usepackage{unicode-math}
        \setmainfont{TeX Gyre Termes}
        \setsansfont{TeX Gyre Heros}
    ''',
}