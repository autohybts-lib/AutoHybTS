Persistência e evidências
=========================

Modelos
-------

.. code-block:: python

   from autohybts import load_model, read_manifest, save_model

   path = save_model(final, "artifacts/champion")
   read_manifest(path)                  # versões, hashes, corte de treino, spec
   model = load_model(path, trusted=True)

O artefato é um diretório com o modelo serializado e um manifesto com checksum.
``load_model`` exige ``trusted=True`` porque desserializar pickle executa código: carregue
só arquivos de origem confiável.

O checksum do manifesto detecta corrupção, não adulteração (os dois arquivos podem ser
trocados juntos). Para isso, assine o artefato com uma chave guardada fora dele ou fixe o
SHA-256 recebido por um canal confiável:

.. code-block:: python

   path = save_model(final, "artifacts/champion", signing_key=key)   # key: 32 bytes secretos
   model = load_model(path, trusted=True, signing_key=key)
   model = load_model(path, trusted=True, sha256="<sha-256 publicado>")

Evidências do experimento
-------------------------

.. code-block:: python

   exp.export_evidence("artifacts/evidence")

Grava a configuração, as divisões, o ledger da busca, as decisões e a auditoria da
seleção — sem valores brutos da série nem do teste selado.

.. seealso::

   Tutoriais: :doc:`../tutorials/notebooks/03_finalize_and_persist`.
