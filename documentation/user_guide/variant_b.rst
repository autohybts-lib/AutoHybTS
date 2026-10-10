Variante B: um passo com parâmetros congelados
==============================================

Na variante A, o modelo prevê ``fh`` passos de uma vez a partir da origem. Na
**variante B**, ele prevê um passo, recebe a observação real e prevê o próximo — sem
reajuste: os parâmetros ficam congelados e só o estado (contexto) avança.

.. code-block:: python

   exp = AutoHybridExperiment(fh=7, variant="B").fit(y)
   result = exp.compare_models()
   final = exp.finalize_model(result.best)

   stream = final.stream()
   ticket = stream.predict_next()
   stream.observe(valor_observado)          # revela o valor; nada é reajustado

* O resíduo observado ``y − b`` só entra na janela do corretor depois da observação.
* ``FinalizedForecaster.update_state`` devolve um novo objeto e recusa ``refit=True``.
* A decomposição ``ma`` funciona em B (é recalculada causalmente); a ``stl`` não.
* A seleção dinâmica do corretor (``dref``, ``local``) só existe aqui: veja
  :doc:`strategies`.
