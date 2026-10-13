Referência da API
=================

Experimento
-----------

.. autoclass:: autohybts.AutoHybridExperiment
   :members: fit, splits, models, register_model, create_model, create_ensemble, create_hybrid,
             create_decomposition,
             create_combination, tune_model, compare_models, finalize_model, predict_model,
             evaluate_external, cross_validate, compare_external, nested_cross_validate,
             data_profile, seasonality_report, characterize, plot_data, plot_model, plot_characterization, get_metrics, export_evidence

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
   :members: name, label, stages, options, describe, ensemble, corrected, depth, model_names

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

.. automodule:: autohybts.strategies.ensemble
   :members: EnsembleForecaster

.. autoclass:: autohybts.registry.capabilities.Capabilities
   :no-members:

.. automodule:: autohybts.domain.fit_count
   :members: count_fits, record_fit

.. automodule:: autohybts.strategies.multistage
   :members: stack_stages

.. automodule:: autohybts.reporting.strategies
   :members: complexity, pareto_front, strategy_table

Modelos próprios
----------------

``AutoHybridExperiment.register_model`` (acima) registra o modelo no experimento.

.. autofunction:: autohybts.models.torch_model
.. autoclass:: autohybts.models.TorchForecaster
.. autoclass:: autohybts.models.TorchCorrector
.. autofunction:: autohybts.models.keras_model
.. autoclass:: autohybts.models.KerasForecaster
.. autoclass:: autohybts.models.KerasCorrector
.. autoclass:: autohybts.models.CustomForecaster
.. autoclass:: autohybts.models.RegressorForecaster
.. autofunction:: autohybts.models.require
.. autofunction:: autohybts.models.check_specifier
.. autofunction:: autohybts.models.validate_requirements
.. autofunction:: autohybts.models.framework_versions

Ablação
-------

.. automodule:: autohybts.benchmarks.ablation
   :members: run_ablation, run_case, ablation_report, ablation_uncertainty,
             ablation_protocol, synthetic_cases, benchmark_cases, AblationCase

Calibração da regra de aceite
-----------------------------

.. automodule:: autohybts.benchmarks.calibration
   :members: run_calibration, calibration_report, calibration_protocol, control_cases,
             ControlCase

Exceções
--------

.. automodule:: autohybts.exceptions
   :members:
