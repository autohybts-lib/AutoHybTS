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
