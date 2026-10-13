About
=====

The ergonomics are inspired by `PyCaret <https://pycaret.org>`_: one experiment object,
short methods and results ready for tables and plots. PyCaret is not a dependency.
AutoHybTS focuses on bringing together, in a single workflow and with leak-free temporal
validation, **residual hybrid** models for time series forecasting and the strategies
for building them — additive correction, multiple stages, decomposition and
combination — accepting complexity only when it pays off in validation.

Principles
----------

* Temporal validity before automation.
* Guarded incremental selection: when in doubt, the simpler model stays.
* Traceability: stable identities, a complete ledger, decisions with reasons.

License and authorship
----------------------

AutoHybTS is free software under the MIT license (``LICENSE`` file in the repository), by
João Ataíde. Code and issues: https://github.com/jvataidee/AutoHybTS.

How to cite
-----------

Cite the version you used (``autohybts.__version__``):

.. code-block:: bibtex

   @software{autohybts,
     author  = {Ataíde, João},
     title   = {{AutoHybTS}: Automated Hybrid Time Series Forecasting},
     year    = {2026},
     version = {1.0.0rc2},
     url     = {https://github.com/jvataidee/AutoHybTS}
   }

For results, also record the seed (``session_id``), the experiment parameters and the
selection snapshot (``result.snapshot`` or ``export_evidence``).

Issues and suggestions
----------------------

Open a GitHub issue with the version, a minimal example and the full error message. For
vulnerabilities, follow the repository's security policy (``.github/SECURITY.md``).
