Ler os resultados
=================

``compare_models`` devolve um ``CompareResult``. Esta página diz o que há em cada tabela e
como ler uma decisão.

Leaderboard
-----------

``result.leaderboard``: uma linha por candidata avaliada na busca (``V_search``),
ordenada pela primeira métrica (menor é melhor).

``name``, ``model``
   nome curto da estrutura (``arima-mlp``) e rótulo completo, com janela, α e fonte de
   resíduos.
``family``, ``strategy``, ``base``, ``corrector``, ``window``, ``alpha``, ``residual_source``
   a configuração.
``mean_rank``
   posto da candidata pela métrica de ordenação (empates com posto médio).
métricas (``MASE``, ``MAE`` …)
   médias sobre as unidades válidas.
``coverage``, ``n_units``, ``n_failed``, ``n_skipped``
   fração das unidades com métrica válida, unidades esperadas e unidades que falharam ou
   foram puladas.
``fit_seconds``, ``predict_seconds``
   custo somado nas origens.
``is_champion``
   marca o campeão final.

``result.selection_leaderboard`` tem as mesmas colunas para as finalistas em
``V_select``.

Decisões
--------

``result.decisions``: uma linha por família que desafiou o campeão, na ordem da cadeia.

``champion_before``, ``challenger``
   os ``candidate_id`` do campeão daquele momento e do desafiante.
``accepted``
   se o desafiante virou campeão.
``median_gain``
   ganho mediano pareado, ``(E_campeão − E_desafiante) / |E_campeão|``, sobre as mesmas
   unidades; positivo quando o desafiante erra menos.
``win_rate``
   fração das unidades em que o desafiante errou menos (empates não contam).
``variability_status``, ``variability_value``
   o critério de variabilidade dos corretores: ``passed``, ``failed``, ``undefined``,
   ``not_applicable`` (desafiante sem corretor) ou ``disabled`` (perfil ``quickstart``).
``p_value``
   o teste de Wilcoxon, só com ``significance_alpha``.
``reason_codes``
   por que a decisão foi tomada:

   .. list-table::
      :header-rows: 1
      :widths: 34 66

      * - Código
        - Significado
      * - ``accepted``
        - todos os critérios passaram
      * - ``median_gain_below_threshold``
        - ganho mediano abaixo de 5 %
      * - ``win_rate_below_threshold``
        - vitórias abaixo de 60 % das unidades
      * - ``coverage_incomplete``
        - o desafiante não tem métrica válida em todas as unidades
      * - ``different_units`` / ``no_comparable_units``
        - as unidades não coincidem ou nenhuma é comparável
      * - ``variability_below_threshold``
        - o corretor quase não mexe na previsão (``correction_std_ratio`` < 0,05)
      * - ``variability_undefined``
        - o critério de variabilidade não pôde ser calculado
      * - ``not_significant``
        - o teste de Wilcoxon não rejeitou (com ``significance_alpha``)

Exemplo real, do notebook 16 (orçamento ``smoke``, ``fold=3``):

.. code-block:: text

   family    champion_before  challenger    accepted  median_gain  win_rate
   linear    arima            linear        False     -0.400       0.000
   kernel    arima            knn           False     -0.357       0.000
   residual  arima            arima-linear  False      0.028       0.667

O ``linear`` e o ``knn`` perderam em todas as unidades. O ``arima-linear`` venceu em 67 %
das unidades, acima dos 60 % exigidos, mas com ganho mediano de 2,8 %, abaixo dos 5 %:
recusado por ``median_gain_below_threshold``, e o ``arima`` continuou campeão.

``result.selection.decision_units_frame()`` desce ao nível de cada unidade: os dois erros,
o ganho e a vitória, para recalcular a regra ou reamostrar fora da biblioteca.

Ledger e auditoria
------------------

``result.search.ledger_frame()``: toda proposta da busca, avaliada ou não, com a etapa
(``S0``–``S4``), a situação e o motivo.

``EVALUATED``
   avaliada em todas as origens de ``V_search``.
``PRUNED_BUDGET``
   cortada pelo orçamento antes de ser avaliada.
``SKIPPED_CAPABILITY``
   o modelo não pode rodar nessa configuração (extra ausente, variante incompatível,
   série sem sazonalidade …); o motivo diz qual.
``DUPLICATE``
   mesma ``candidate_id`` de uma proposta anterior.
``STOPPED``
   não começou porque o prazo ``max_wall_time_s`` acabou.

``result.audit`` resume o quanto a seleção foi adaptativa (tentativas, cortes, finalistas,
motivos). ``result.audit.selection_gap()`` compara o escore do vencedor em ``V_search``
com o de ``V_select``: uma diferença grande indica otimismo de escolher onde se procurou.

Métricas
--------

Toda métrica é "menor é melhor" e é calculada por origem, sobre os passos ``1..H``. Uma
métrica indefinida numa unidade vale ``NaN``: a unidade fica inválida para ela e reduz a
cobertura, em vez de entrar no ranking com um valor arbitrário. Escalas abaixo de
``1e-12`` contam como zero.

.. list-table::
   :header-rows: 1
   :widths: 14 56 30

   * - Métrica
     - Fórmula
     - Indefinida quando
   * - ``MAE``
     - ``mean(|y − ŷ|)``
     - —
   * - ``RMSE``
     - ``sqrt(mean((y − ŷ)²))``
     - —
   * - ``sMAPE``
     - ``mean(2|y − ŷ| / (|y| + |ŷ|))``, fração em [0, 2]; 0/0 conta como 0
     - —
   * - ``MASE`` (padrão)
     - ``MAE / mean(|y_t − y_{t−m}|)`` no treino da origem, ``m`` = período sazonal
     - escala nula ou treino com até ``m`` pontos
   * - ``RMSSE``
     - ``sqrt(mean((y − ŷ)²) / mean((y_t − y_{t−m})²))`` no treino da origem
     - escala nula ou treino com até ``m`` pontos
   * - ``WAPE``
     - ``sum(|y − ŷ|) / sum(|y|)``
     - ``sum(|y|)`` nula
   * - ``RRMSE``
     - ``RMSE / mean(y)`` nos passos ``1..H``
     - ``mean(y)`` nula
   * - ``RMAE``
     - ``MAE / mean(y)`` nos passos ``1..H``
     - ``mean(y)`` nula

* A escala da MASE e da RMSSE vem só do treino de cada origem, nunca dos alvos.
* Em demanda intermitente, prefira ``RMSSE``: métricas de erro absoluto favorecem prever
  zero.
* ``metrics=[...]`` escolhe as métricas; a primeira ordena e decide.

.. seealso::

   :doc:`selection` (a regra de aceite) e :doc:`search_space` (etapas e ledger).
