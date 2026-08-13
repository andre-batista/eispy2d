Installation
============

Eispy2d can be installed either directly from the Python Package Index
(PyPI) or from the source code hosted on GitHub.

Requirements
------------

Eispy2d requires:

- Python 3.9 or newer
- ``pip``

Method 1: Install via pip (Recommended)
--------------------------------------

The recommended way to install Eispy2d is through PyPI:

.. code-block:: bash

   pip install eispy2d

This command installs the library and all required dependencies
automatically.

Verify the Installation
-----------------------

After installation, you can verify that everything is working by opening
a Python interpreter and importing the package:

.. code-block:: python

   from eispy2d.core import configuration
   from eispy2d.solvers.inverse import bim

If these imports run without errors, Eispy2d is installed correctly.

Method 2: Install from Source
-----------------------------

If you want to inspect the source code or contribute to development,
clone the repository and install the dependencies manually:

.. code-block:: bash

   git clone https://github.com/andre-batista/eispy2d.git
   cd eispy2d
   pip install -r requirements.txt

Development Installation (Optional)
-----------------------------------

To make local changes immediately available in your Python environment,
install the package in editable mode:

.. code-block:: bash

   pip install -e .

Upgrade to the Latest Version
-----------------------------

To update an existing installation from PyPI:

.. code-block:: bash

   pip install --upgrade eispy2d

Usage Examples
--------------

The repository contains scripts and Jupyter notebooks in the ``demo``
directory demonstrating how to:

- Build a scattering problem
- Configure inverse solvers
- Run case studies
- Execute benchmarks
- Analyze reconstruction results

See the :doc:`examples` section for detailed walkthroughs.

Troubleshooting
---------------

If installation fails:

1. Ensure you are using a supported Python version.
2. Upgrade ``pip``:

   .. code-block:: bash

      pip install --upgrade pip

3. Reinstall the package:

   .. code-block:: bash

      pip install --force-reinstall eispy2d