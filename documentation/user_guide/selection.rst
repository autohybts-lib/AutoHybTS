Seleção incremental
===================

A seleção começa no melhor modelo individual e visita as famílias na ordem
``family_order``: estatística, linear, kernel, árvore, neural, residual, decomposição,
combinação e seleção dinâmica. A finalista de cada família — sua melhor estrutura na
busca — desafia o **campeão do momento** em ``V_select``.

Algumas famílias só entram na cadeia quando são usadas, e o protocolo padrão não muda
sem elas:

* ``intermittent`` (``croston_sba``, ``tsb``, ``adida``, ``imapa``), quando um desses
  modelos está em ``base_models``, e as famílias de modelos registrados com
  ``register_model``, assim que registrados: logo antes de ``residual``;
* ``ensemble``, quando incluída em ``compare_models(include=...)``: por último, antes de
  ``dynamic``.

Famílias sem finalista (não pedidas no ``include`` ou sem candidata válida) são puladas.

Regra de aceite
---------------

Com os parâmetros padrão, o desafiante substitui o campeão só se, nas mesmas unidades (origens):

* tiver cobertura completa e métricas finitas;
* **ganho mediano** pareado ``(E_campeão − E_desafiante) / |E_campeão|`` ≥ 5 %;
* **taxa de vitórias** ≥ 60 % (empates não contam);
* **variabilidade** ``correction_std_ratio`` ≥ 0,05 quando houver corretor — um
  corretor que quase não mexe na previsão é inerte. O valor é a mediana por série
  (sobre as unidades comparadas) e depois a mediana entre as séries; uma série sem
  diagnóstico finito aparece como ``NaN`` em ``variability_by_series`` da decisão e
  fica fora da mediana (versão 2 do critério).

Opcionalmente, ``AutoHybridExperiment(significance_alpha=0.05)`` acrescenta um critério
estatístico: o desafiante também precisa vencer num teste de Wilcoxon pareado unilateral
nas mesmas unidades, ``p < alfa`` (motivo ``not_significant``; o p-valor aparece na
coluna ``p_value`` das decisões). Unidades com erros iguais saem do teste; com ``n``
unidades restantes (todas as séries juntas) o menor p-valor possível é ``1/2ⁿ``: são
precisas ao menos 5 para ``alfa = 0,05``. Com ``fold=3`` numa série, o critério
**nunca** passa. O teste controla cada comparação, não a
cadeia inteira: várias famílias desafiam um campeão que muda com as decisões
anteriores, e origens vizinhas não são independentes.

Calibração da regra
~~~~~~~~~~~~~~~~~~~

Os limiares e o número de origens vêm de uma calibração com resposta conhecida
(``autohybts.benchmarks.calibration``): 50 séries de ruído branco, onde
qualquer estrutura composta é um falso aceite, e 50 com resíduo AR(0,8), onde a
correção é útil. A meta, fixada antes de medir, era falso aceite ≤ 10 % e ganho útil
aceito ≥ 50 %, por cadeia.

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Regra
     - Falso aceite
     - Ganho útil aceito
   * - ganho ≥ 1 %, ``fold=3`` (padrão anterior)
     - 38 %
     - 66 %
   * - ganho ≥ 5 %, ``fold=3``
     - 16 %
     - 60 %
   * - ganho ≥ 5 %, ``fold=5``
     - 12 %
     - 72 %
   * - **ganho ≥ 5 %, ``fold=7``** (padrão)
     - **2 %** (confirmação: 4 %)
     - **54 %** (confirmação: 56 %)

Por isso ``fold="auto"`` (padrão do experimento) usa 7 origens por camada quando o
histórico permite. Numa série curta usa o maior número que cabe e avisa: a regra
continua mais conservadora que a anterior, mas a taxa de falso aceite medida vale só
para 7 unidades. Um ``fold`` inteiro é respeitado como dado; o número usado fica em
``exp.fold_``. Num painel, as unidades de todas as séries entram juntas na comparação.

Rejeição mantém o campeão. Cada decisão fica registrada, aceita ou não
(``result.selection.decision_units_frame()`` traz, por decisão e unidade, os dois
erros, o ganho e a vitória, para recalcular ou reamostrar fora da biblioteca):

.. code-block:: python

   result.decisions
   #   family    champion_before  challenger   accepted  median_gain  win_rate ...
   #   residual  arima            arima-mlp    False     -0.07        0.33

Perfis
------

``quickstart``
   critério de variabilidade desligado por padrão (e dito explicitamente).
``standard``
   regra completa (padrão).
``research_strict``
   regra completa e cobertura total obrigatória.
``exploratory``
   resultados marcados como exploratórios na auditoria.

Ranking
-------

Na busca e entre séries, cada estrutura é representada pela sua melhor configuração;
estruturas são ordenadas por posto (empates com posto médio), depois erro agregado,
cobertura, custo e ``candidate_id``. Num painel, os postos são calculados por série e
promediados.

.. warning::

   Sem ``significance_alpha``, a regra é um protocolo, não um teste de hipótese: em ruído
   branco ela aceitou híbridos espúrios em parte das sementes testadas. O número de
   origens em ``V_select`` pesa muito: com 2 unidades, 60 % de vitórias exige vencer as
   duas, o que acontece por acaso em cerca de 1 de cada 4 comparações. Meça o
   comportamento da cadeia com ``autohybts.benchmarks.calibration`` e use a comparação
   e a CV aninhada para relatar a incerteza.

.. seealso::

   Tutoriais: :doc:`../tutorials/notebooks/02_hybrid_selection`, :doc:`../tutorials/notebooks/09_ablation`, :doc:`../tutorials/notebooks/11_calibration`.

Limiares configuráveis
----------------------

Os parâmetros públicos abaixo também funcionam em ``space`` no JSON da ablação.
Os defaults permanecem: ganho mediano 0,05, vitórias 0,60, cobertura completa e epsilon
1e-12. ``variability_threshold=None`` herda o perfil (0,05, ou desligado em
``quickstart``); um valor numérico explícito liga o critério, inclusive em
``quickstart``. ``research_strict`` continua exigindo cobertura completa.

.. code-block:: python

   exp = AutoHybridExperiment(
       median_gain_threshold=0.05, win_rate_threshold=0.60,
       variability_threshold=0.05, min_coverage=1.0, epsilon=1e-12,
   )

Os limiares devem ser números finitos, sem booleanos. Ganho e variabilidade são
não negativos; vitórias ficam em [0, 1], cobertura em (0, 1] e epsilon é positivo.
Os valores efetivos entram na política, nos snapshots e no hash do protocolo de
seleção. Propagam-se às buscas aninhadas e ao ``AutoHybridForecaster``. Sua alteração
não muda o protocolo da busca nem acessa o teste selado. Fixe-os antes da avaliação
externa; a calibração dos defaults não certifica valores customizados.
