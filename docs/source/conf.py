# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

import apax

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("../../"))
sys.path.insert(0, os.path.abspath("_ext"))

project = "Apax"
copyright = "2025, Moritz Schäfer, Nico Segreto, Fabian Zills, Johannes Kästner"
author = "Moritz Schäfer, Nico Segreto, Fabian Zills, Johannes Kästner"
release = apax.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "nbsphinx",
    "nbsphinx_link",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinx_design",
    "config_tables",
]

myst_enable_extensions = ["colon_fence", "deflist", "fieldlist"]
myst_heading_anchors = 3

templates_path = ["_templates"]
exclude_patterns = ["reference/_generated/*"]  # included from reference pages


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"

html_title = f"apax {release}"
html_theme_options = {
    "source_repository": "https://github.com/apax-hub/apax/",
    "source_branch": "main",
    "source_directory": "docs/source/",
}
