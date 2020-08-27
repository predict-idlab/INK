import os
import sys
sys.path.insert(0, os.path.abspath('../..'))  # Source code dir relative to this file

extensions = [
    'sphinx.ext.autodoc',  # Core library for html generation from docstrings
    'sphinx.ext.autosummary',  # Create neat summary tables
]
autosummary_generate = True  # Turn on sphinx.ext.autosummary



# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']


project = 'INK'
copyright = '2020, Bram Steenwinckel'
author = 'Bram Steenwincinckel'
version = ''
release = '0.0.2'
templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'
pygments_style = 'sphinx'
#html_theme = 'classic'
