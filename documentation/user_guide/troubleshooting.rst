Solução de problemas
====================

"fold='auto': o histórico comporta N origem(ns) por camada"
-----------------------------------------------------------

A série é curta demais para 7 origens por camada. O experimento usa o maior número que
cabe (``exp.fold_``) e avisa porque a regra de aceite foi calibrada para 7: com menos
unidades, ela aceita estruturas espúrias com mais frequência. Use um ``fh`` menor, uma
série mais longa ou leia as decisões com cautela; um ``fold`` inteiro silencia o aviso,
mas não muda o risco.

Um modelo aparece como ``SKIPPED``
----------------------------------

O motivo está no ledger (``result.search.ledger_frame()``, coluna ``reason``) ou na
mensagem de ``create_model``. Os casos comuns:

* extra ausente: a mensagem traz o comando ``pip install "autohybts[...]"``;
* modelo incompatível com a variante B (não incorpora observações sem reajuste);
* série sem sazonalidade declarada (``seasonal_naive``, ``sarima``, STL com período 1);
* capacidades fora da busca: modelo global, saída não pontual ou horizonte máximo menor
  que o pedido.

A busca segue com os outros modelos; só "nenhuma candidata válida" interrompe a execução.

``DataValidationError`` no ``fit``
----------------------------------

A biblioteca não reordena, não reamostra, não remove duplicatas e não preenche faltantes
em silêncio. A mensagem diz o que corrigir:

* valores ausentes (``NaN``): trate-os antes, com o critério do seu domínio;
* datas duplicadas ou fora de ordem: agregue ou ordene a série;
* frequência irregular ou lacunas nas datas: reindexe a série na frequência certa e
  decida como preencher; ou informe ``frequency``.

Meu híbrido foi recusado
------------------------

Veja ``result.decisions``: ``reason_codes`` diz qual critério falhou e
``median_gain``/``win_rate`` mostram por quanto (:doc:`results`). Recusar é o
comportamento esperado quando o ganho não é robusto: o modelo mais simples continua.
Não ajuste limiares para fazer um híbrido passar; se precisar medir a regra, use
``autohybts.benchmarks.calibration`` (:doc:`benchmarks`).

A busca demora
--------------

O custo é proporcional a candidatas × origens × ajustes por candidata
(:doc:`search_space`). Para explorar rápido: ``budget="smoke"``, ``fold=3`` (com a ressalva
da calibração) e menos ``base_models``/``correctors``. ``n_jobs`` avalia candidatas em
paralelo sem mudar o resultado. ``max_wall_time_s`` limita o tempo, mas torna o resultado
dependente da máquina. Modelos caros: ``sarima`` em séries diárias longas, ``AutoARIMA``,
redes na CPU (o ``MambaNet`` em especial).

O resultado mudou de uma máquina para outra
-------------------------------------------

Com a mesma série, a mesma ``session_id`` e os mesmos parâmetros, a busca é a mesma.
Diferenças vêm de: ``max_wall_time_s`` (o prazo corta propostas diferentes), GPU ou
TensorFlow (não determinismo residual) e versões diferentes das bibliotecas. O manifesto
de um modelo salvo (``read_manifest``) registra as versões usadas.

``SealedTestAccessError`` ao avaliar de novo
--------------------------------------------

``evaluate_external`` lê o teste selado uma única vez por modelo finalizado. Repetir a
leitura transformaria o teste em validação. Guarde o resultado da primeira chamada
(``external = exp.evaluate_external(final)``); para comparar vários modelos no teste, use
``exp.compare_external()``, que avalia cada um uma vez.

``predict_model`` recusa vários passos na variante B
----------------------------------------------------

Um modelo da variante B prevê um passo por vez. Use ``fh=1`` ou ``final.stream()``, que
alterna ``predict_next`` e ``observe`` (:doc:`variant_b`).

``load_model`` exige ``trusted=True``
-------------------------------------

Carregar um modelo desserializa pickle, o que executa código. ``trusted=True`` declara que
o arquivo é de origem confiável; para verificar a integridade, passe ``sha256=`` (o hash
recebido por um canal confiável) ou ``signing_key=`` (:doc:`persistence`).

StatsForecast e pandas 3
------------------------

As versões atuais do StatsForecast exigem pandas < 3. Instale o extra ``nixtla-stats``
num ambiente separado (:doc:`nixtla`); sem ele, os modelos ``sf_*`` ficam ``SKIPPED``.
