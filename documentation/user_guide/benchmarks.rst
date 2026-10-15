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

Ganhos não finitos (``NaN`` ou infinito) não entram nas frações de melhora, empate e
piora. O relatório informa ``n_rows``, ``n_paired``, ``n_invalid_gain`` e
``paired_coverage`` sobre as linhas fornecidas; esses valores não representam casos
planejados que não chegaram à tabela. A coluna ``coverage`` continua sendo a cobertura
de validação. Na incerteza, ``n_series`` conta as séries presentes, e
``n_paired_series``, ``n_invalid_series`` e ``paired_series_coverage`` identificam as
séries com ganho finito após agregar as sementes válidas. Sem pares válidos, estimativas
e intervalos ficam indefinidos, em vez de serem apresentados como zero.

``ablation_protocol`` registra as métricas, a política inicial e os limites de orçamento
da configuração recebida em ``space``, incluindo perfil, significância e sobrescritas
de orçamento. O número de origens resolvido por série e a ordem de famílias efetiva
pertencem aos snapshots da comparação; gerar esse protocolo não executa buscas nem
abre o teste externo. O ganho mediano do relatório agrega linhas série × semente,
enquanto o intervalo agrega sementes dentro de cada série antes de reamostrar.

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

Referência individual e versão do procedimento
------------------------------------------------------------

O esquema ``ablation/3`` usa ``lowest_select_error`` tanto na referência individual
da CV aninhada quanto no recuo de ``guarded_nested``: o finalista individual sem
exógenas de menor erro válido em ``V_select``, com cobertura suficiente. As linhas
registram ``standalone_reference_rule``, ``best_base_id`` e ``initial_champion_id``.
O campeão inicial segue a ordem da busca e pode ser diferente dessa referência.
O esquema anterior ``ablation/2`` usava o campeão inicial dentro da CV e outra regra
no recuo. Os resultados dos dois esquemas avaliam procedimentos diferentes e devem
ser identificados separadamente; não basta alterar o rótulo dos arquivos anteriores.

Protocolo configurável e retomada
------------------------------------------------------------

A CLI aceita um arquivo JSON com esquema ``autohybts-ablation-config/1``. O exemplo
``tools/protocols/ablation_demo.json`` usa dados sintéticos e orçamento de demonstração:

.. code-block:: bash

   python -m autohybts.benchmarks.ablation --config tools/protocols/ablation_demo.json --checkpoint artifacts/demo.sqlite --out artifacts/demo.csv
   python -m autohybts.benchmarks.ablation --config tools/protocols/ablation_demo.json --checkpoint artifacts/demo.sqlite --resume --out artifacts/demo.csv

``preset="demo"`` mantém duas origens por camada e orçamento ``smoke``. O preset
``research`` usa ``fold="auto"`` e orçamento ``small``; é um ponto de configuração,
não um protocolo científico aprovado ou uma garantia de sete origens em séries curtas.
``space`` aceita os parâmetros do experimento, como modelos, fontes residuais, grades,
métricas, perfil e número de origens. Horizonte, tamanho do teste e identificação dos
dados vêm dos casos; não podem ser sobrescritos por ``space``. O teste oficial dos
benchmarks é preservado. O período sazonal do descriptor é transportado para o caso;
``seasonal_period`` na raiz do JSON permite uma escolha explícita (``null`` infere).

Para benchmarks, ``series_ids`` fixa a lista, ou ``series`` e ``sampling_seed`` definem
uma amostra. A semente da amostragem é diferente de ``seeds``, que controla os modelos.
Não combine IDs explícitos com uma contagem, nem ``--config`` com alterações científicas
na CLI. Sem JSON, ``--fold``, ``--sampling-seed`` e ``--series-ids`` também estão disponíveis.
Solicitar mais séries que as disponíveis gera erro, em vez de reduzir a amostra em silêncio.

Na API Python, use os mesmos casos, sementes e parâmetros:

.. code-block:: python

   table = run_ablation(
       synthetic_cases(2), seeds=(0, 1), space={"fold": "auto"},
       checkpoint="artifacts/study.sqlite", on_error="record",
   )
   table = run_ablation(
       synthetic_cases(2), seeds=(0, 1), space={"fold": "auto"},
       checkpoint="artifacts/study.sqlite", resume=True, on_error="record",
   )
   table.attrs["coverage"]   # denominador planejado e estados dos casos
   table.attrs["cases"]      # tentativas, motivos, tempos e snapshots por caso
   table.attrs["protocol"]   # configuração, dados, ambiente e identidade do código

O checkpoint SQLite salva atomicamente o conjunto dos quatro procedimentos de cada
série × semente. Casos ``SUCCEEDED`` não são executados novamente. Casos interrompidos
são reexecutados por inteiro na retomada; ``FAILED`` e ``SKIPPED`` só são repetidos com ``--retry-failed`` ou
``retry_failed=True``, junto de retomada explícita. O padrão continua sendo parar em
erro; ``--on-error record`` permite continuar após falhas de caso. Erros de configuração
e vazamento temporal sempre interrompem a execução.

O hash do estudo inclui configuração, dados (valores e datas), versões e fontes Python
instaladas, inclusive alterações não commitadas. Uma incompatibilidade impede a retomada.
Um lock do sistema operacional admite um escritor por arquivo e é liberado também após
encerramento do processo; o arquivo ``.lock`` pode permanecer no disco. Não são gravados
valores brutos das séries. O checkpoint não alimenta busca nem seleção.

As fases ``full`` e ``simple`` registram snapshots resolvidos, e ``nested`` conserva os
snapshots das buscas internas, divisões efetivas, erros e custos por dobra. Tentativas
interrompidas mantêm seu histórico; após encerramento abrupto, a duração desconhecida
fica ``null``. A tabela CSV contém resultados concluídos; os casos que falharam permanecem
no checkpoint e no ``<out>.protocol.json``, junto da cobertura e dos snapshots. Esse JSON
é necessário para reconstruir o denominador, que o CSV de sucessos sozinho não contém.
``coverage.procedures`` separa, para cada procedimento, casos planejados, concluídos,
com métrica válida e com ganho pareado válido. Concluir uma execução não implica que
todas as métricas estejam definidas. Valores indefinidos são ``null`` no JSON e ``NaN``
na tabela reconstruída.
``n_attempts`` e ``known_seconds_total`` incluem tentativas anteriores; durações perdidas
após encerramento abrupto são contadas em ``n_unknown_duration_attempts``. Snapshots
parciais de tentativas anteriores são preservados quando um caso é repetido.

Piloto de custo somente no desenvolvimento
------------------------------------------

Em um checkout dos fontes, ``tools/protocols/m4_daily_development_pilot.json`` oferece
um piloto limitado do fluxo: D1/D2/D3, uma semente, orçamento smoke, variante A e até
os últimos 730 pontos de cada série de treino oficial. ``development_only=true`` lê
somente metadados e treino do M4; o arquivo de teste oficial não é baixado nem lido.
O bloco de teste relatado são os últimos 14 pontos do treino retido, identificados
como ``development_tail`` na proveniência de cada caso.

.. code-block:: bash

   # Opcional: defina AUTOHYBTS_DATA_DIR para um cache dedicado.
   python tools/audit_development_pilot.py --config tools/protocols/m4_daily_development_pilot.json --outdir artifacts/pilot/audit --plot
   python -m autohybts.benchmarks.ablation --config tools/protocols/m4_daily_development_pilot.json --checkpoint artifacts/pilot/state.sqlite --out artifacts/pilot/results.csv --on-error record

A auditoria confere valores finitos, timestamps regulares/únicos/ordenados e os planos
temporais principal e interno mais antigo, sem treinar modelos. As datas seguem o
índice de origem ou sintético do provedor; a regularidade não certifica o calendário
original. Cada caso registra a auditoria, hashes dos arquivos de origem, comprimento
original e o limite explícito ``development_tail``. A CLI também grava um CSV de
auditoria antes do treinamento quando ``--out`` é informado.

``fold="auto"`` usa até 7 origens por camada. Nesta configuração D1/D2 comportam 7;
a D3, mais curta, comporta 3 e sua busca interna 2. Os avisos e planos resolvidos
permanecem visíveis. Isso não valida a calibração padrão em históricos curtos. O
piloto mede funcionamento e custo sob uma configuração limitada; não constitui o
benchmark oficial M4 nem comparação representativa de desempenho. Congele outro
protocolo antes do estudo principal e não escolha seus parâmetros pelo teste oficial.

Para carregar treino diretamente, ``M4Provider.load(..., with_test=False)`` retorna
sem partição de teste. O default ``with_test=True`` preserva o comportamento anterior.
