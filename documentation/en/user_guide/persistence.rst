Persistence and evidence
========================

Models
------

.. code-block:: python

   from autohybts import load_model, read_manifest, save_model

   path = save_model(final, "artifacts/champion")
   read_manifest(path)                  # versions, hashes, training cutoff, spec
   model = load_model(path, trusted=True)

The artifact is a directory with the serialized model and a manifest with a checksum.
``load_model`` requires ``trusted=True`` because unpickling executes code: only load
files from a trusted source.

The manifest checksum detects corruption, not tampering (both files can be swapped
together). For that, sign the artifact with a key kept outside it, or pin the SHA-256
received through a trusted channel:

.. code-block:: python

   path = save_model(final, "artifacts/champion", signing_key=key)   # key: 32 secret bytes
   model = load_model(path, trusted=True, signing_key=key)
   model = load_model(path, trusted=True, sha256="<published sha-256>")

Experiment evidence
-------------------

.. code-block:: python

   exp.export_evidence("artifacts/evidence")

Writes the configuration, the splits, the search ledger, the decisions and the
selection audit — without raw values of the series or of the sealed test.

.. seealso::

   Tutorials: :doc:`../tutorials/notebooks/03_finalize_and_persist`.

Evidence for an article
-----------------------

``export_evidence`` writes selection decisions and paired units, search costs in the
trial ledger, dated splits and the resolved snapshot. Supply already computed results
to include external metrics and nested folds, costs and snapshots. Exporting never
evaluates or refits models; the caller associates the supplied results with the run.

.. code-block:: python

   external = exp.evaluate_external(exp.finalize_model(result.best))
   nested = exp.nested_cross_validate(outer_folds=2)
   exp.export_evidence(
       "artifacts/evidence-new", external=(external,), nested=nested,
       plots=("selection_trace",), figure_formats=("png", "pdf", "svg"),
   )

Use an empty directory. The ``autohybts-evidence/2`` manifest records SHA-256 per file,
optional dependency versions, a source fingerprint including uncommitted changes,
and separate run, export and protocol IDs. The run ID stays fixed across exports
after the same ``fit``. Version-1 bundles remain readable as ordinary files, without
the new integrity fields. Raw values are excluded by default; explicitly setting
``include_predictions=True`` adds external forecasts and observations, flagged in
the manifest. Figures may reveal values; use ``selection_trace`` to share only
selection indicators.
