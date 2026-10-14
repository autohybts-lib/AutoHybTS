Avaliar o próprio procedimento
==============================

``autohybts.benchmarks`` mede o próprio procedimento de seleção. Nada aqui é importado
por ``import autohybts``.

Ablação
-------

Quatro procedimentos com o mesmo orçamento e os mesmos testes externos.

=======================  ============================================================
Procedimento             O que faz
=======================  ============================================================
``lowest_error``         a finalista de menor erro em ``V_select``, sem limiares
``incremental``          a cadeia incremental guardada, todas as famílias
``incremental_simple``   a mesma cadeia, só estruturas com até 2 modelos
``guarded_nested``       como ``incremental``; se a CV aninhada (mesmas famílias)
                         disser que o procedimento não supera a melhor base da
                         seleção interna, fica a melhor base de ``V_select``
=======================  ============================================================

.. code-block:: python

   from autohybts.benchmarks import (
       ablation_report, ablation_uncertainty, run_ablation, synthetic_cases,
   )

   table = run_ablation(synthetic_cases(12), seeds=(0, 1, 2))
   ablation_report(table)                  # por procedimento
   ablation_report(table, by="stratum")    # por estrato (previsibilidade, subconjunto)
   ablation_uncertainty(table)             # intervalo reamostrando séries

.. code-block:: bash

   python -m autohybts.benchmarks.ablation --dataset synthetic --series 12 --seeds 0 1 2
   python -m autohybts.benchmarks.ablation --dataset m4-daily --series 20 --out ablation.csv

Com ``--out``, o protocolo resolvido (ambiente, commit, orçamento, espaço, sementes e
séries) vai para ``<out>.protocol.json``. Alterações locais não commitadas não são
detectadas: registre-as à parte.

Cada linha (série × semente × procedimento) traz o erro externo do escolhido, o da
melhor base de ``V_select`` e o das referências ``naive`` e ``seasonal_naive``, e o
custo completo do procedimento:

=====================  ==============================================================
Coluna                 Significado
=====================  ==============================================================
``search``             busca usada (``full`` é compartilhada por três procedimentos)
``candidates``         candidatas avaliadas, incluindo as buscas internas da CV
                       aninhada no ``guarded_nested``
``units``              unidades (origem × série) avaliadas na busca, na seleção e na
                       CV aninhada; cada unidade é um modelo novo
``seconds``            custo do procedimento: busca + seleção (+ CV aninhada)
``seconds_*``          por fase; ``seconds_refit`` (reajuste antes do teste) é relato
``stop_reason``        motivo de parada da busca
``max_trials_total``   orçamento configurado
=====================  ==============================================================

O relatório traz, por procedimento: ganho mediano contra a melhor base no teste,
frações de casos melhores, iguais e piores que ela, fração de híbridos e complexidade
média, lacuna validação–teste, posto médio, dispersão entre sementes, cobertura e custo.
``ablation_uncertainty`` resume cada série (mediana entre sementes) antes de reamostrar
séries: sementes de uma mesma série não são unidades independentes.

.. warning::

   O teste externo só relata: não escolha o procedimento "vencedor" com ele.

.. note::

   Os presets de orçamento limitam só o número de candidatas, então a mesma série e a
   mesma semente dão a mesma busca em qualquer máquina. Um prazo de relógio
   (``max_wall_time_s``) é opcional; com ele, as candidatas avaliadas podem mudar entre
   máquinas, por isso a ablação não o usa.

O notebook ``09_ablation`` traz uma execução completa (12 séries sintéticas × 3
sementes, orçamento ``small``) com tabelas e figuras.

Calibração da regra de aceite
-----------------------------

``autohybts.benchmarks.calibration`` mede a regra em controles com resposta conhecida:

* ``white_noise`` (controle negativo): nível mais ruído branco; um híbrido campeão é um
  **falso aceite** de complexidade;
* ``ar_residual`` (controle positivo): tendência, sazonalidade semanal e ruído AR(0,8),
  com bases que não modelam o AR; um campeão individual é um **ganho útil recusado**.

Para cada série e número de origens em ``V_select`` (``fold``), uma busca e uma seleção
rodam uma vez; cada política (a regra, a regra com Wilcoxon …) é aplicada aos mesmos
finalistas. As taxas saem por cadeia (o campeão final) e por comparação, com intervalos
que reamostram séries. Os conjuntos ``calibration`` e ``confirmation`` usam sementes
disjuntas: ajuste numa, confira uma vez na outra. O teste externo não é lido.

.. code-block:: bash

   python -m autohybts.benchmarks.calibration --series 50 --folds 2 3 5 --out cal.csv
   python -m autohybts.benchmarks.calibration --series 50 --split confirmation --out conf.csv

O notebook ``11_calibration`` roda os dois controles nos dois conjuntos, com tabelas e
um gráfico das taxas por número de origens.

.. seealso::

   Tutoriais: :doc:`../tutorials/notebooks/05_benchmark_m4`, :doc:`../tutorials/notebooks/09_ablation`,
   :doc:`../tutorials/notebooks/17_benchmark_study`.
