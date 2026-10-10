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
