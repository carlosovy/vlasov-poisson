
"""Sphinx configuration for the Vlasov-Poisson documentation."""

project = "Vlasov-Poisson"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
]

# Support Markdown and reStructuredText.
source_suffix = [".rst", ".md"]

# Interpret NumPy-style docstrings.
napoleon_numpy_docstring = True
napoleon_google_docstring = False

# Generate individual API pages with autosummary.
autosummary_generate = True

# Include documented public members in the API reference.
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
}

# Use the standard Sphinx HTML theme.
html_theme = "pydata_sphinx_theme"

# Exclude generated documentation from source scanning.
exclude_patterns = ["build"]
