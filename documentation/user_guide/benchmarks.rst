Ablação do procedimento
=======================

``autohybts.benchmarks`` mede o próprio procedimento de seleção: quatro
procedimentos com o mesmo orçamento e os mesmos testes externos.

=======================  ============================================================
Procedimento             O que faz
=======================  ============================================================
``lowest_error``         a finalista de menor erro em ``V_select``, sem limiares
``incremental``          a cadeia incremental guardada, todas as famílias
``incremental_simple``   a mesma cadeia, só estruturas com até 2 modelos
``guarded_nested``       como ``incremental``; se a CV aninhada disser que o
                         procedimento não supera a melhor base, fica a base
=======================  ============================================================

.. code-block:: python

   from autohybts.benchmarks import ablation_report, run_ablation, synthetic_cases

   table = run_ablation(synthetic_cases(12), seeds=(0, 1, 2))
   ablation_report(table)

.. code-block:: bash

   python -m autohybts.benchmarks.ablation --dataset synthetic --series 12 --seeds 0 1 2
   python -m autohybts.benchmarks.ablation --dataset m4-daily --series 20 --out ablation.csv

O relatório traz, por procedimento: ganho mediano contra a melhor base no teste, fração
de séries com híbrido escolhido e com campeão melhor que a base, lacuna validação–teste,
posto médio, dispersão entre sementes, cobertura, tempo e número de candidatas.

.. warning::

   O teste externo só relata: não escolha o procedimento "vencedor" com ele.

.. note::

   Os presets de orçamento têm limite de tempo por busca (60 s no ``smoke``). Numa
   máquina ocupada, buscas podem parar pelo relógio e mudar os números entre execuções;
   para comparações reprodutíveis, use um orçamento com folga de tempo.

O notebook ``09_ablation`` traz uma execução completa (12 séries sintéticas × 3
sementes) com tabelas e figuras.
