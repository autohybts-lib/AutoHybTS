Modelos próprios: PyTorch, TensorFlow e outras bibliotecas
==========================================================

Um modelo montado fora da AutoHybTS entra no experimento por **registro**. Depois disso
ele é usado pelo nome em tudo: ``create_model``, busca (``base_models``), corretor de
híbridos (``correctors``), estágios, decomposição, combinação, seleção dinâmica,
``tune_model``, painéis, variante B, finalização e persistência — com as mesmas regras
de validação temporal.

.. code-block:: python

   from autohybts import AutoHybridExperiment
   from autohybts.models import torch_model
   from autohybts.models.torch_zoo import LSTMNet

   exp = AutoHybridExperiment(fh=14, base_models=["linear", "lstm"], correctors=["linear"])
   exp.register_model(
       "lstm",
       torch_model(LSTMNet, hidden=32, lags=28, differencing=1),
       family="deep",
       requires={"torch": ">=2.2"},   # versão exigida da biblioteca
       version="1",                   # identidade desta registração
   )
   result = exp.fit(y).compare_models()

Instalação dos extras: ``pip install "autohybts[torch]"`` ou
``pip install "autohybts[tensorflow]"``. ``import autohybts`` nunca carrega essas
bibliotecas; elas são importadas quando o modelo é ajustado.

Quatro maneiras de trazer o modelo
----------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Wrapper
     - Contrato
   * - ``torch_model(Modulo, **padrões)``
     - ``Modulo(n_inputs=..., **params)`` devolve um ``torch.nn.Module``; o ``forward``
       recebe ``(batch, n_inputs)`` com os lags do mais antigo ao mais recente e devolve
       ``(batch,)``
   * - ``keras_model(builder, **padrões)``
     - ``builder(n_inputs=..., **params)`` devolve um ``keras.Model`` **não compilado**
       de ``(batch, n_inputs)`` para ``(batch, 1)`` (TensorFlow 2.16+/Keras 3)
   * - ``CustomForecaster(fit_fn, predict_fn)``
     - ``fit_fn(y, **params)`` devolve um estado; ``predict_fn(estado, horizonte,
       histórico, **params)`` devolve as previsões. Serve para qualquer biblioteca
   * - ``RegressorForecaster(estimador)``
     - qualquer objeto com ``fit(X, y)``/``predict(X)`` sobre os lags (LightGBM,
       XGBoost, CatBoost, scikit-learn)

Nos dois primeiros, os parâmetros do wrapper (``lags``, ``differencing``, ``scale``,
``epochs``, ``batch_size``, ``learning_rate``, ``validation_fraction``, ``patience``,
``random_state``, ``device``) vão para o wrapper e os demais para o módulo ou builder.
Todos podem ser mudados por candidata (``create_model("lstm", hidden=64)``) e ajustados
no ``tune_model``.

Com ``kind="corrector"`` a mesma rede vira corretor de resíduos: recebe a janela de
resíduos (``window`` do híbrido) e prevê o próximo resíduo.

.. code-block:: python

   exp.register_model("lstm_corr", torch_model(LSTMNet, kind="corrector", hidden=8),
                      kind="corrector", requires={"torch": ">=2.2"})
   exp.create_hybrid("seasonal_naive", "lstm_corr")

O que o wrapper garante
-----------------------

* **Sem vazamento:** cada origem recebe um modelo novo; diferenciação e padronização
  são ajustadas só no vetor de treino daquela origem; a parada antecipada usa as
  **últimas linhas do próprio treino** (``validation_fraction``), nunca ``V_select`` nem
  o teste.
* **Pesos congelados na previsão:** a previsão de vários passos é recursiva; um contexto
  mais longo (variante B, resíduos fora da amostra) só muda as entradas.
* **Semente:** ``random_state`` fixa a semente do PyTorch ou do Keras; o mesmo modelo
  nos mesmos dados dá a mesma previsão na CPU.
* **Versão exigida:** ``requires`` é conferido quando o modelo é construído para uma
  tentativa; biblioteca ausente ou antiga torna a tentativa ``SKIPPED`` com o motivo e
  o comando ``pip`` — o resto do experimento segue.
* **Identidade:** ``version`` da registração entra na chave de cada tentativa (cache e
  snapshot). Mude-a quando mudar o código do modelo.
* **Persistência:** ``save_model`` guarda os pesos (``state_dict`` do PyTorch, pesos do
  Keras) e o manifesto registra as versões das bibliotecas. Para carregar em outro
  processo, o módulo ou builder precisa ser importável (definido num ``.py``), não numa
  célula de notebook.

GPU
---

Os wrappers do PyTorch aceitam ``device="cpu"`` (padrão), ``"cuda"`` ou ``"auto"`` (GPU
quando existe). Instale a build do PyTorch com CUDA compatível com o driver, por exemplo
``pip install torch --index-url https://download.pytorch.org/whl/cu130``, e confira com
``torch.cuda.is_available()``. O TensorFlow **não** usa GPU no Windows nativo desde a
versão 2.11 (só via WSL2 ou Linux); no Windows as redes Keras rodam na CPU.

Com redes pequenas e janelas curtas, a GPU nem sempre ganha: o custo de disparar cada
operação domina. Medido numa RTX 3050 Laptop (``synthetic_daily``, 500 pontos, 14 lags,
60 épocas, todas as origens de um experimento):

.. list-table::
   :header-rows: 1
   :widths: 25 18 18 18 18

   * - Rede
     - CPU, lote 32
     - GPU, lote 32
     - CPU, lote 128
     - GPU, lote 128
   * - ``LSTMNet``
     - 40 s
     - 17 s
     - 7 s
     - 6 s
   * - ``TransformerNet``
     - 32 s
     - 32 s
     - 11 s
     - 11 s
   * - ``MambaNet``
     - 65 s
     - 23 s
     - 61 s
     - 12 s

Lotes maiores aceleram, mas fazem menos atualizações por época e podem piorar o ajuste;
confira o erro. Com a mesma semente, CPU e GPU dão números parecidos, não idênticos.

Família e seleção
-----------------

``family`` (padrão ``"deep"``) é a família do modelo na cadeia incremental. Uma família
nova desafia o campeão logo antes dos híbridos residuais; uma família existente
(``"neural"``, ``"tree"`` ...) entra no lugar dela. Escolha ``"deep"`` para comparar as
redes entre si e decidir uma vez se a família vale a complexidade.

Arquiteturas de exemplo
-----------------------

``autohybts.models.torch_zoo`` (importa o PyTorch):

* ``MLPNet``: camadas densas sobre o vetor de lags;
* ``LSTMNet``: LSTM sobre os lags como sequência;
* ``TransformerNet``: encoder Transformer com posições aprendidas e máscara causal;
* ``MambaNet``: bloco de espaço de estados seletivo no estilo Mamba (Gu e Dao, 2023) em
  PyTorch puro, com varredura sequencial. Não é o pacote otimizado ``mamba-ssm``, que
  exige CUDA no Linux; para as janelas curtas de modelos de lags a varredura simples
  basta.

``autohybts.models.keras_zoo``: ``mlp``, ``lstm`` e ``transformer``.

.. tip::

   Em séries com tendência, use ``differencing=1``: redes não extrapolam bem além da
   faixa vista no treino. Na série ``synthetic_daily`` (500 pontos, horizonte 14), uma
   LSTM pequena foi de MASE 2,56 sem diferenciação para 0,80 com ela (o linear fez 0,92).

O notebook ``10_external_models`` traz o fluxo completo com PyTorch (LSTM, Transformer,
Mamba e um GRU escrito na hora), Keras (LSTM, Transformer e uma convolução escrita na
hora), Holt-Winters do statsmodels por funções e gradient boosting do scikit-learn.
