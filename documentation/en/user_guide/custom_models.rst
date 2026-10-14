Your own models: PyTorch, TensorFlow and other libraries
========================================================

A model built outside AutoHybTS enters the experiment through **registration**. After
that it is used by name everywhere: ``create_model``, search (``base_models``), hybrid
corrector (``correctors``), stages, decomposition, combination, dynamic selection,
``tune_model``, panels, variant B, finalization and persistence — with the same
temporal validation rules.

.. code-block:: python

   from autohybts import AutoHybridExperiment
   from autohybts.models import torch_model
   from autohybts.models.torch_zoo import LSTMNet

   exp = AutoHybridExperiment(fh=14, base_models=["linear", "lstm"], correctors=["linear"])
   exp.register_model(
       "lstm",
       torch_model(LSTMNet, hidden=32, lags=28, differencing=1),
       family="deep",
       requires={"torch": ">=2.2"},   # required library version
       version="1",                   # identity of this registration
   )
   result = exp.fit(y).compare_models()

Install from PyPI: ``python -m pip install "autohybts[torch]==1.0.0rc3"`` or
``python -m pip install "autohybts[tensorflow]==1.0.0rc3"``. ``import autohybts`` never loads these
libraries; they are imported when the model is fitted.

Four ways to bring the model in
-------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Wrapper
     - Contract
   * - ``torch_model(Module, **defaults)``
     - ``Module(n_inputs=..., **params)`` returns a ``torch.nn.Module``; ``forward``
       receives ``(batch, n_inputs)`` with the lags from oldest to most recent and
       returns ``(batch,)``
   * - ``keras_model(builder, **defaults)``
     - ``builder(n_inputs=..., **params)`` returns an **uncompiled** ``keras.Model``
       from ``(batch, n_inputs)`` to ``(batch, 1)`` (TensorFlow 2.16+/Keras 3)
   * - ``CustomForecaster(fit_fn, predict_fn)``
     - ``fit_fn(y, **params)`` returns a state; ``predict_fn(state, horizon, history,
       **params)`` returns the forecasts. Works for any library
   * - ``RegressorForecaster(estimator)``
     - any object with ``fit(X, y)``/``predict(X)`` on the lags (LightGBM, XGBoost,
       CatBoost, scikit-learn)

In the first two, the wrapper parameters (``lags``, ``differencing``, ``scale``,
``epochs``, ``batch_size``, ``learning_rate``, ``validation_fraction``, ``patience``,
``random_state``, ``device``) go to the wrapper and the others to the module or
builder. All of them can be changed per candidate (``create_model("lstm", hidden=64)``)
and tuned in ``tune_model``.

With ``kind="corrector"`` the same network becomes a residual corrector: it receives the
residual window (the hybrid's ``window``) and forecasts the next residual.

.. code-block:: python

   exp.register_model("lstm_corr", torch_model(LSTMNet, kind="corrector", hidden=8),
                      kind="corrector", requires={"torch": ">=2.2"})
   exp.create_hybrid("seasonal_naive", "lstm_corr")

What the wrapper guarantees
---------------------------

* **No leakage:** each origin gets a fresh model; differencing and standardization are
  fitted only on that origin's training vector; early stopping uses the **last rows of
  the training data itself** (``validation_fraction``), never ``V_select`` or the test.
* **Separate internal validation:** the standardization mean and deviation come only
  from the internal training rows, before the tail reserved for early stopping; the
  tail is scaled with those statistics and does not change what the network learns. The
  trial diagnostics include ``train_rows`` and ``validation_rows``. On series with a
  strong trend, use ``differencing=1``: without differencing, the tail falls outside the
  range seen in training.
* **Frozen weights at prediction time:** multi-step forecasting is recursive; a longer
  context (variant B, out-of-sample residuals) only changes the inputs.
* **Seed:** ``random_state`` fixes the PyTorch or Keras seed; the same model on the same
  data gives the same forecast on the CPU (numerical tolerance of ``1e-5``; GPU and
  TensorFlow may have residual non-determinism). Because these seeds are process-global,
  network training is serialized across threads: with ``n_jobs > 1`` the other
  candidates keep running in parallel and each network gets the same weights it would
  get alone.
* **Required version:** ``requires`` follows PEP 440 (``">=2.2,<3"``, ``"~=2.2"``,
  ``"!=2.3.*"``), checked with the ``packaging`` package (included in the ``torch`` and
  ``tensorflow`` extras). A malformed specifier is refused at registration
  (``ConfigurationError``). The installed version is checked when the model is built for
  a trial; a missing library or one out of range makes the trial ``SKIPPED`` with the
  reason and the ``pip`` command — the rest of the experiment goes on. Pre-releases come
  before the final version: ``2.2.0rc1`` does not satisfy ``>=2.2``, but ``2.3.0rc1``
  does; ``2.2.0.post1`` is not ``==2.2`` (use ``==2.2.*``); local labels like
  ``+cu130`` are ignored by specifiers without a local label.
* **Identity:** the registration's ``version`` enters the key of each trial (cache and
  snapshot). Change it when you change the model's code.
* **Persistence:** ``save_model`` stores the weights (PyTorch ``state_dict``, Keras
  weights) and the manifest records the library versions. To load in another process,
  the module or builder must be importable (defined in a ``.py`` file), not in a
  notebook cell.

GPU
---

The PyTorch wrappers accept ``device="cpu"`` (default), ``"cuda"`` or ``"auto"`` (GPU
when available). Install the PyTorch build with a CUDA version compatible with the
driver, for example
``pip install torch --index-url https://download.pytorch.org/whl/cu130``, and check
with ``torch.cuda.is_available()``. TensorFlow does **not** use the GPU on native
Windows since version 2.11 (only through WSL2 or Linux); on Windows, Keras networks run
on the CPU.

With small networks and short windows, the GPU does not always win: the cost of
launching each operation dominates. Measured on 13 October 2026 on an RTX 3050 Laptop,
with PyTorch 2.14.1+cu130 (``synthetic_daily``, 500 points, 14 lags, 60 epochs, the 14
origins of an experiment with ``fold="auto"``; time of ``create_model``):

.. list-table::
   :header-rows: 1
   :widths: 25 18 18 18 18

   * - Network
     - CPU, batch 32
     - GPU, batch 32
     - CPU, batch 128
     - GPU, batch 128
   * - ``LSTMNet``
     - 54 s
     - 30 s
     - 17 s
     - 7 s
   * - ``TransformerNet``
     - 172 s
     - 159 s
     - 68 s
     - 39 s
   * - ``MambaNet``
     - 1523 s
     - 95 s
     - 1251 s
     - 78 s

``MambaNet`` is slow on the CPU because the vectorized scan computes every pair of steps
in the window at once (quadratic cost in the number of lags): that avoids a Python loop
and favours the GPU, but multiplies the work on the CPU. Use it with a GPU. Larger
batches are faster but make fewer updates per epoch and can hurt the fit; check the
error. With the same seed, CPU and GPU give similar, not identical, numbers.

Family and selection
--------------------

``family`` (default ``"deep"``) is the model's family in the incremental chain. A new
family challenges the champion right before the residual hybrids; an existing family
(``"neural"``, ``"tree"`` ...) takes that family's place. Choose ``"deep"`` to compare
the networks with each other and decide once whether the family is worth the
complexity.

Example architectures
---------------------

``autohybts.models.torch_zoo`` (imports PyTorch):

* ``MLPNet``: dense layers on the lag vector;
* ``LSTMNet``: LSTM over the lags as a sequence;
* ``TransformerNet``: Transformer encoder with learned positions and a causal mask;
* ``MambaNet``: Mamba-style selective state-space block (Gu and Dao, 2023) in plain
  PyTorch. The selective scan computes every step at once (cumulative sums in log scale
  with a causal mask), without a Python loop, and gives the same result as the
  step-by-step recursion. It is not the optimized ``mamba-ssm`` package, which requires
  CUDA on Linux; for the short windows of lag models this version is enough.

``autohybts.models.keras_zoo``: ``mlp``, ``lstm`` and ``transformer``.

.. tip::

   On series with a trend, use ``differencing=1``: networks do not extrapolate well
   beyond the range seen in training. On the ``synthetic_daily`` series (500 points,
   horizon 14), a small LSTM went from MASE 2.56 without differencing to 0.80 with it
   (linear got 0.92).

The ``10_external_models`` notebook has the complete workflow with PyTorch (LSTM,
Transformer, Mamba and a GRU written on the spot), Keras (LSTM, Transformer and a
convolution written on the spot), statsmodels' Holt-Winters through functions and
scikit-learn gradient boosting.

.. seealso::

   Tutorials: :doc:`../tutorials/notebooks/10_external_models`.
