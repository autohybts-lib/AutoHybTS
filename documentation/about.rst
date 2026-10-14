Sobre
=====

A ergonomia é inspirada no `PyCaret <https://pycaret.org>`_: um objeto de experimento,
métodos curtos e resultados prontos para tabela e gráfico. O PyCaret não é dependência.
O foco da AutoHybTS é reunir, num só fluxo e com validação temporal sem vazamento, os
modelos **híbridos residuais** para previsão de séries temporais e as estratégias de
montá-los — correção aditiva, vários estágios, decomposição e combinação —, aceitando
complexidade só quando ela se paga na validação.

Princípios
----------

* Validade temporal antes de automação.
* Seleção incremental guardada: na dúvida, fica o modelo mais simples.
* Rastreabilidade: identidades estáveis, ledger completo, decisões com motivo.

Licença e autoria
-----------------

AutoHybTS é software livre sob a licença MIT (arquivo ``LICENSE`` do repositório), de
João Ataíde. Código e issues: https://github.com/jvataidee/AutoHybTS.

Como citar
----------

Cite a versão que você usou (``autohybts.__version__``):

.. code-block:: bibtex

   @software{autohybts,
     author  = {Ataíde, João},
     title   = {{AutoHybTS}: Automated Hybrid Time Series Forecasting},
     year    = {2026},
     version = {1.0.0rc3},
     url     = {https://github.com/jvataidee/AutoHybTS}
   }

Para resultados, registre também a semente (``session_id``), os parâmetros do experimento
e o snapshot da seleção (``result.snapshot`` ou ``export_evidence``).

Problemas e sugestões
---------------------

Abra uma issue no GitHub com a versão, um exemplo mínimo e a mensagem de erro completa.
Para vulnerabilidades, siga a política de segurança do repositório
(``.github/SECURITY.md``).
