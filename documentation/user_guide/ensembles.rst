Ensembles e composições
=======================

Um *ensemble* combina **previsões de candidatas completas**: modelos isolados, híbridos
residuais ou outros ensembles. Uma *composição corrigida* aplica um corretor residual
sobre uma candidata composta. As duas formas aceitam aninhamento até três níveis, por
exemplo um ensemble cujo membro é um híbrido sobre outro ensemble.

.. code-block:: python

   linear = exp.create_model("linear")
   ensemble = exp.create_ensemble(["seasonal_naive", linear, "knn"], method="nnls")
   corrected = exp.create_hybrid(ensemble, "linear")      # ensemble + correção residual
   of_hybrids = exp.create_ensemble(
       [exp.create_hybrid("naive", "linear"), exp.create_hybrid("seasonal_naive", "knn")]
   )

Agregações
----------

==================  ===========================================================================
``method``          O que faz
==================  ===========================================================================
``equal``           média; pesos ``1/K``
``median``          mediana por passo (3 ou mais membros)
``trimmed_mean``    descarta ``max(1, round(trim·K))`` previsões de cada ponta por passo e faz a
                    média do resto (``trim``, padrão 0,2; 3 ou mais membros)
``inverse_mse``     pesos ``∝ 1/MSE`` dos erros fora da amostra de um passo
``nnls``            mínimos quadrados não negativos sobre as previsões fora da amostra,
                    normalizados e encolhidos para os pesos iguais:
                    ``w = (1 − shrinkage)·w_nnls + shrinkage/K`` (padrão 0,5)
==================  ===========================================================================

Pesos aprendidos usam só o vetor de treino de cada origem: cada membro é reajustado no
prefixo e prevê a cauda um passo por vez; depois todos são ajustados no vetor inteiro.
Com poucas origens, pesos livres tendem a piorar frente à média simples — por isso o
encolhimento. Os pesos aparecem nos diagnósticos de cada tentativa (``weight_i``, na
ordem dos membros do rótulo).

Na busca
--------

.. code-block:: python

   result = exp.compare_models(include=("standalone", "residual", "ensemble"))

Com ``"ensemble"``, a etapa S4 propõe (família ``ensemble``, com ``ensemble_methods``):

* ensembles da melhor base de cada família (até 3 membros);
* ensembles das duas melhores estruturas residuais (agregações ponderadas);
* o ensemble de pesos iguais corrigido por cada corretor configurado.

A versão de pesos iguais de cada tipo vem primeiro, para que um corte de orçamento
mantenha o controle mais simples. A família ``ensemble`` desafia o campeão por último
(antes de ``dynamic``), só depois das estruturas mais simples, e passa pela mesma regra
de aceite: um ensemble que não supera o campeão é recusado com os motivos registrados.

.. warning::

   ``"ensemble"`` fica fora do ``include`` padrão até sua taxa de aceite espúrio ser
   calibrada (:doc:`benchmarks`, ``--include standalone residual ensemble``). Cada
   família nova é mais um desafiante na cadeia e aumenta a chance de aceitar
   complexidade por acaso.

Custo
-----

Cada tentativa conta seus ajustes de modelos-folha (``TrialResult.n_fits``;
``result.search.fit_count``). Um ensemble ``nnls`` de ``K`` membros ajusta ``2K``
modelos por origem (prefixo e vetor inteiro); um híbrido, três (base, base no prefixo e
corretor). Compare ganhos sempre com o custo.

.. seealso::

   Tutoriais: :doc:`../tutorials/notebooks/13_ensembles`.
