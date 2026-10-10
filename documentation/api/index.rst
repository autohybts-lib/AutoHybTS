Referência da API
=================

Experimento
-----------

.. autoclass:: autohybts.AutoHybridExperiment
   :members: fit, splits, models, create_model, create_hybrid, create_decomposition,
             create_combination, tune_model, compare_models, finalize_model, predict_model,
             evaluate_external, cross_validate, compare_external, nested_cross_validate,
             data_profile, plot_data, plot_model, get_metrics, export_evidence

.. autoclass:: autohybts.AutoHybridForecaster
   :members:

Dados
-----

.. autoclass:: autohybts.PanelFrame
   :members: single, from_dict, from_long, series_ids, frequency, to_long

.. autofunction:: autohybts.datasets.get_data
.. autofunction:: autohybts.datasets.list_datasets
.. autofunction:: autohybts.datasets.load_benchmark
.. autofunction:: autohybts.datasets.load_dataset
.. autofunction:: autohybts.datasets.raw_dataset
.. autofunction:: autohybts.datasets.describe_dataset

.. autoclass:: autohybts.datasets.BenchmarkSeries
   :members: full
.. autoclass:: autohybts.datasets.BenchmarkDataset
   :members: series_ids, series, panel
.. autoclass:: autohybts.datasets.DatasetDescriptor
.. autoclass:: autohybts.datasets.CacheManager
   :members: fetch, verify

Modelos finalizados e persistência
----------------------------------

.. autoclass:: autohybts.FinalizedForecaster
   :members: predict, stream, update_state, training_cutoff, last_observation

.. autoclass:: autohybts.PanelForecaster
   :members: predict, series_ids, training_cutoff

.. autofunction:: autohybts.save_model
.. autofunction:: autohybts.load_model
.. autofunction:: autohybts.read_manifest

Resultados
----------

.. autoclass:: autohybts.results.CompareResult
   :members: strategy_table, strategies, plot

.. autoclass:: autohybts.results.ModelResult
   :members:

.. autoclass:: autohybts.results.ExternalResult
.. autoclass:: autohybts.results.PredictResult

.. autoclass:: autohybts.reporting.comparison.ComparisonResult
   :members: summary, win_rates, tests, friedman, plot

Especificações
--------------

.. autoclass:: autohybts.domain.specs.CandidateSpec
   :members: name, label, stages, options, describe

.. autoclass:: autohybts.domain.specs.ModelSpec
   :members: of, params, label

Busca, seleção e ajuste
-----------------------

.. autoclass:: autohybts.search.spaces.SearchSpace
.. autoclass:: autohybts.search.budgets.SearchGuardConfig
   :members: preset
.. autoclass:: autohybts.selection.policy.SelectionPolicy
.. autoclass:: autohybts.search.tuning.Distribution
.. autoclass:: autohybts.search.tuning.TuningResult
   :members: improvement

Estratégias
-----------

.. automodule:: autohybts.strategies.decomposition
   :members: DecompositionForecaster, causal_moving_average, kurtosis_window

.. automodule:: autohybts.strategies.combination
   :members: FixedCombinationForecaster

.. automodule:: autohybts.strategies.multistage
   :members: stack_stages

.. automodule:: autohybts.reporting.strategies
   :members: complexity, pareto_front, strategy_table

Ablação
-------

.. automodule:: autohybts.benchmarks.ablation
   :members: run_ablation, run_case, ablation_report, synthetic_cases, benchmark_cases,
             AblationCase

Exceções
--------

.. automodule:: autohybts.exceptions
   :members:
