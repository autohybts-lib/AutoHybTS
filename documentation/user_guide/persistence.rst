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

Evidências para o artigo
------------------------

``export_evidence`` grava decisões e unidades pareadas, custos da busca no ledger,
splits datados e o snapshot resolvido. Passe resultados já calculados para incluir
métricas externas e dobras, custos e snapshots aninhados. Exportar não avalia nem
reajusta modelos; cabe a quem chama associar os resultados fornecidos à execução.

.. code-block:: python

   external = exp.evaluate_external(exp.finalize_model(result.best))
   nested = exp.nested_cross_validate(outer_folds=2)
   exp.export_evidence(
       "artifacts/evidence-nova", external=(external,), nested=nested,
       plots=("selection_trace",), figure_formats=("png", "pdf", "svg"),
   )

Use uma pasta vazia. O manifesto ``autohybts-evidence/2`` registra SHA-256 por arquivo,
versões de dependências opcionais, hash dos fontes incluindo alterações sem commit
e IDs distintos de execução, exportação e protocolo. O ID de execução se mantém nas
exportações após o mesmo ``fit``. Pacotes de versão 1 continuam legíveis como arquivos,
sem os novos campos de integridade. Valores brutos ficam fora por padrão;
``include_predictions=True`` inclui explicitamente previsões e observações externas,
assinaladas no manifesto. Figuras podem revelar valores; use ``selection_trace`` para
compartilhar apenas indicadores de seleção.
