Tutoriais
=========

Notebooks curtos, um tema cada, no estilo dos tutoriais do PyCaret. Estão em
``notebooks/`` no repositório, já com as saídas, que aparecem aqui.

* :doc:`notebooks/01_quickstart`: o fluxo completo numa série, com figuras.
* :doc:`notebooks/02_hybrid_selection`: como a busca monta os híbridos e por que um desafiante é aceito ou recusado.
* :doc:`notebooks/03_finalize_and_persist`: finalizar, avaliar uma vez no teste selado, salvar, carregar e prever em produção.
* :doc:`notebooks/04_rolling_one_step`: variante B em streaming, base e correção a cada passo, A × B no mesmo trecho.
* :doc:`notebooks/05_benchmark_m4`: M4, M3 e Rdatasets com o teste oficial selado (baixa dados na primeira execução).
* :doc:`notebooks/06_compare_and_cross_validate`: comparação estatística de finalistas e validação cruzada temporal.
* :doc:`notebooks/07_panel`: várias séries no mesmo experimento.
* :doc:`notebooks/08_advanced_families_and_tuning`: estágios, decomposição, combinação e ajuste de hiperparâmetros.
* :doc:`notebooks/09_ablation`: ablação do procedimento de seleção (pesado: cerca de 1 h).
* :doc:`notebooks/10_external_models`: modelos de PyTorch, TensorFlow/Keras e outras bibliotecas (pesado).
* :doc:`notebooks/11_calibration`: calibração da regra de aceite em controles com resposta conhecida.
* :doc:`notebooks/12_baselines_and_boosting`: baselines, boosting e corretores opcionais.
* :doc:`notebooks/13_ensembles`: ensembles de modelos e de híbridos.
* :doc:`notebooks/14_intermittent_demand`: demanda intermitente: Croston, TSB, ADIDA, IMAPA.
* :doc:`notebooks/15_statsforecast`: modelos automáticos do StatsForecast (ambiente com pandas 2.x).
* :doc:`notebooks/16_series_diagnostics`: sazonalidade, estacionariedade, diferenças e caracterização.
* :doc:`notebooks/17_benchmark_study`: 17 séries clássicas em três classes, cinco modelos (com o híbrido ARIMA–ANN), protocolos A e B, 3, 6 e 12 lags e horizontes até 12 (baixa dados na primeira execução).

.. toctree::
   :hidden:

   notebooks/01_quickstart
   notebooks/02_hybrid_selection
   notebooks/03_finalize_and_persist
   notebooks/04_rolling_one_step
   notebooks/05_benchmark_m4
   notebooks/06_compare_and_cross_validate
   notebooks/07_panel
   notebooks/08_advanced_families_and_tuning
   notebooks/09_ablation
   notebooks/10_external_models
   notebooks/11_calibration
   notebooks/12_baselines_and_boosting
   notebooks/13_ensembles
   notebooks/14_intermittent_demand
   notebooks/15_statsforecast
   notebooks/16_series_diagnostics
   notebooks/17_benchmark_study
