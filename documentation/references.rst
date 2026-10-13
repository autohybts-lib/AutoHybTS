Referências
===========

Artigos e livros que definem os métodos usados ou discutidos na AutoHybTS, com o link
para o texto (o DOI, quando existe). Dentro de cada seção, a ordem é cronológica. A linha
*Na AutoHybTS* diz onde o método aparece na biblioteca; *Contexto* marca trabalhos que
não estão implementados, mas explicam de onde vem uma ideia ou o que a biblioteca
deliberadamente não faz.

Autores, ano, veículo, volume e páginas foram conferidos no Crossref (outubro de 2026);
os itens sem DOI, no site do periódico ou dos anais.

Revisões
--------

* **De Gooijer, J. G., & Hyndman, R. J. (2006).** 25 years of time series forecasting.
  *International Journal of Forecasting*, 22(3), 443–473.
  `doi:10.1016/j.ijforecast.2006.01.001 <https://doi.org/10.1016/j.ijforecast.2006.01.001>`__

  *Contexto:* panorama de 1982 a 2005 — suavização exponencial, ARIMA, modelos não
  lineares, combinação e medidas de acurácia.

* **Hajirahimi, Z., & Khashei, M. (2019).** Hybrid structures in time series modeling
  and forecasting: A review. *Engineering Applications of Artificial Intelligence*, 86,
  83–106.
  `doi:10.1016/j.engappai.2019.08.018 <https://doi.org/10.1016/j.engappai.2019.08.018>`__

  *Contexto:* revisão das estruturas híbridas em previsão de séries temporais.

* **Petropoulos, F., Apiletti, D., Assimakopoulos, V., et al. (2022).** Forecasting:
  theory and practice. *International Journal of Forecasting*, 38(3), 705–871.
  `doi:10.1016/j.ijforecast.2021.11.001 <https://doi.org/10.1016/j.ijforecast.2021.11.001>`__

  *Contexto:* visão geral escrita por mais de 80 autores, de modelos estatísticos a
  aprendizado de máquina, combinação e avaliação.

* **Wang, X., Hyndman, R. J., Li, F., & Kang, Y. (2023).** Forecast combinations: An
  over 50-year review. *International Journal of Forecasting*, 39(4), 1518–1547.
  `doi:10.1016/j.ijforecast.2022.11.005 <https://doi.org/10.1016/j.ijforecast.2022.11.005>`__

  *Contexto:* cinquenta anos de combinação de previsões, de Bates e Granger (1969) aos
  pesos aprendidos.

Combinação de previsões e ensembles
-----------------------------------

* **Bates, J. M., & Granger, C. W. J. (1969).** The combination of forecasts.
  *Operational Research Quarterly* (hoje *Journal of the Operational Research
  Society*), 20(4), 451–468.
  `doi:10.1057/jors.1969.103 <https://doi.org/10.1057/jors.1969.103>`__

  Combinar previsões pode errar menos do que escolher a melhor; pesos inversamente
  proporcionais ao erro passado. *Na AutoHybTS:* pesos ``inverse_mse`` da combinação de
  bases, da combinação de corretores e de ``create_ensemble``
  (:doc:`user_guide/strategies`, :doc:`user_guide/ensembles`).

* **Granger, C. W. J., & Ramanathan, R. (1984).** Improved methods of combining
  forecasts. *Journal of Forecasting*, 3(2), 197–204.
  `doi:10.1002/for.3980030207 <https://doi.org/10.1002/for.3980030207>`__

  A combinação como regressão do valor observado nas previsões. *Na AutoHybTS:* ideia do
  ``nnls`` (regressão sobre previsões fora da amostra).

* **Clemen, R. T. (1989).** Combining forecasts: A review and annotated bibliography.
  *International Journal of Forecasting*, 5(4), 559–583.
  `doi:10.1016/0169-2070(89)90012-5 <https://doi.org/10.1016/0169-2070(89)90012-5>`__

  Consolida a evidência de que combinar, inclusive pela média simples, costuma reduzir o
  erro. *Na AutoHybTS:* ``equal`` (pesos iguais) é o controle proposto primeiro na
  busca.

* **Diebold, F. X., & Pauly, P. (1990).** The use of prior information in forecast
  combination. *International Journal of Forecasting*, 6(4), 503–508.
  `doi:10.1016/0169-2070(90)90028-A <https://doi.org/10.1016/0169-2070(90)90028-A>`__

  Encolhimento dos pesos estimados em direção aos pesos iguais. *Na AutoHybTS:*
  ``shrinkage`` do ``nnls``.

* **Wolpert, D. H. (1992).** Stacked generalization. *Neural Networks*, 5(2), 241–259.
  `doi:10.1016/S0893-6080(05)80023-1 <https://doi.org/10.1016/S0893-6080(05)80023-1>`__

  Um meta-modelo aprende a combinar as previsões fora da amostra dos modelos de base.
  *Na AutoHybTS:* pesos aprendidos e resíduos ``historical_oof_*`` vêm de previsões fora
  da amostra dentro do vetor de treino, nunca de ajustes dentro da amostra.

* **Breiman, L. (1996).** Stacked regressions. *Machine Learning*, 24(1), 49–64.
  `doi:10.1007/BF00117832 <https://doi.org/10.1007/BF00117832>`__

  Stacking com pesos não negativos (mínimos quadrados não negativos) sobre previsões de
  validação cruzada. *Na AutoHybTS:* ``method="nnls"``.

* **Breiman, L. (1996).** Bagging predictors. *Machine Learning*, 24(2), 123–140.
  `doi:10.1007/BF00058655 <https://doi.org/10.1007/BF00058655>`__

  Agregação de modelos ajustados em reamostras bootstrap. *Na AutoHybTS:* base do
  ``random_forest``.

* **Freund, Y., & Schapire, R. E. (1997).** A decision-theoretic generalization of
  on-line learning and an application to boosting. *Journal of Computer and System
  Sciences*, 55(1), 119–139.
  `doi:10.1006/jcss.1997.1504 <https://doi.org/10.1006/jcss.1997.1504>`__

  AdaBoost. *Na AutoHybTS:* origem do ``adaboost_r2`` (Drucker, 1997).

* **Timmermann, A. (2006).** Forecast combinations. In G. Elliott, C. W. J. Granger &
  A. Timmermann (Eds.), *Handbook of Economic Forecasting* (Vol. 1, pp. 135–196).
  Elsevier.
  `doi:10.1016/S1574-0706(05)01004-9 <https://doi.org/10.1016/S1574-0706(05)01004-9>`__

  *Contexto:* teoria dos pesos, instabilidade e por que a média simples é difícil de
  superar.

* **Jose, V. R. R., & Winkler, R. L. (2008).** Simple robust averages of forecasts: Some
  empirical results. *International Journal of Forecasting*, 24(1), 163–169.
  `doi:10.1016/j.ijforecast.2007.06.001 <https://doi.org/10.1016/j.ijforecast.2007.06.001>`__

  Médias aparadas e winsorizadas de previsões. *Na AutoHybTS:* ``median`` e
  ``trimmed_mean``.

* **Smith, J., & Wallis, K. F. (2009).** A simple explanation of the forecast
  combination puzzle. *Oxford Bulletin of Economics and Statistics*, 71(3), 331–355.
  `doi:10.1111/j.1468-0084.2008.00541.x <https://doi.org/10.1111/j.1468-0084.2008.00541.x>`__

  Pesos estimados perdem para a média simples pelo erro de estimação dos próprios pesos.
  *Na AutoHybTS:* motivo do encolhimento do ``nnls`` quando há poucas origens.

* **Albuquerque, R. V. de, & Oliveira, J. F. L. de (2025).** A comparative analysis of
  combination operators in heterogeneous ensembles for one-step-ahead time series
  forecasting. In *Anais do XXII Encontro Nacional de Inteligência Artificial e
  Computacional (ENIAC 2025)* (pp. 61–72). SBC.
  `doi:10.5753/eniac.2025.11788 <https://doi.org/10.5753/eniac.2025.11788>`__

  Compara média, mediana, média ponderada e stacking (SVR) num ensemble heterogêneo.
  *Na AutoHybTS:* as mesmas famílias de agregação de ``create_ensemble``.

Sistemas híbridos residuais
---------------------------

Correção do resíduo
~~~~~~~~~~~~~~~~~~~

* **Zhang, G. P. (2003).** Time series forecasting using a hybrid ARIMA and neural
  network model. *Neurocomputing*, 50, 159–175.
  `doi:10.1016/S0925-2312(01)00702-0 <https://doi.org/10.1016/S0925-2312(01)00702-0>`__

  O ARIMA prevê a série, uma rede neural prevê o resíduo do ARIMA e as duas previsões
  são somadas. *Na AutoHybTS:* ``residual_additive`` (``arima-mlp``,
  ``ŷ = b + α·r̂``; em Zhang, α = 1) — :doc:`user_guide/strategies`.

* **Khashei, M., & Bijari, M. (2011).** A novel hybridization of artificial neural
  networks and ARIMA models for time series forecasting. *Applied Soft Computing*,
  11(2), 2664–2675.
  `doi:10.1016/j.asoc.2010.10.015 <https://doi.org/10.1016/j.asoc.2010.10.015>`__

  *Contexto:* troca a soma por uma combinação não linear da previsão do ARIMA, dos
  resíduos e das defasagens da série. A AutoHybTS mantém a combinação aditiva.

* **Firmino, P. R. A., de Mattos Neto, P. S. G., & Ferreira, T. A. E. (2014).**
  Correcting and combining time series forecasters. *Neural Networks*, 50, 1–11.
  `doi:10.1016/j.neunet.2013.10.008 <https://doi.org/10.1016/j.neunet.2013.10.008>`__

  *Contexto:* corrige um previsor com um modelo dos seus erros e combina as previsões.

* **de Mattos Neto, P. S. G., Ferreira, T. A. E., Lima, A. R., Vasconcelos, G. C., &
  Cavalcanti, G. D. C. (2017).** A perturbative approach for enhancing the performance
  of time series forecasting. *Neural Networks*, 88, 114–124.
  `doi:10.1016/j.neunet.2017.02.004 <https://doi.org/10.1016/j.neunet.2017.02.004>`__

  Cadeia de modelos, cada um ajustado ao erro deixado pelos anteriores. *Na AutoHybTS:*
  ``residual_multistage`` (``arima-mlp-linear``).

* **de Mattos Neto, P. S. G., Cavalcanti, G. D. C., & Madeiro, F. (2017).** Nonlinear
  combination method of forecasters applied to PM time series. *Pattern Recognition
  Letters*, 95, 65–72.
  `doi:10.1016/j.patrec.2017.06.008 <https://doi.org/10.1016/j.patrec.2017.06.008>`__

  *Contexto:* NoLiC — a função que combina previsão e correção é aprendida.

* **Panigrahi, S., & Behera, H. S. (2017).** A hybrid ETS–ANN model for time series
  forecasting. *Engineering Applications of Artificial Intelligence*, 66, 49–59.
  `doi:10.1016/j.engappai.2017.07.007 <https://doi.org/10.1016/j.engappai.2017.07.007>`__

  A estrutura de Zhang com suavização exponencial na base. *Na AutoHybTS:*
  ``create_hybrid("exponential_smoothing", "mlp")``.

* **Silva, E. G., Santos Júnior, D. S. de O., Cavalcanti, G. D. C., & de Mattos Neto,
  P. S. G. (2018).** Improving the accuracy of intelligent forecasting models using the
  perturbation theory. In *2018 International Joint Conference on Neural Networks
  (IJCNN)* (pp. 1–7). IEEE.
  `doi:10.1109/IJCNN.2018.8489697 <https://doi.org/10.1109/IJCNN.2018.8489697>`__

  Correção sucessiva de modelos de aprendizado de máquina. *Na AutoHybTS:* estágios
  extras de ``residual_multistage``.

* **Santos Júnior, D. S. de O., de Oliveira, J. F. L., & de Mattos Neto, P. S. G.
  (2019).** An intelligent hybridization of ARIMA with machine learning models for time
  series forecasting. *Knowledge-Based Systems*, 175, 72–86.
  `doi:10.1016/j.knosys.2019.03.011 <https://doi.org/10.1016/j.knosys.2019.03.011>`__

  *Contexto:* ARIMA, um corretor do resíduo e um combinador não linear aprendido.

* **da Silva, E. G., de Mattos Neto, P. S. G., & de Oliveira, J. F. L. (2019).** Hybrid
  system for time series using iterative residual forecasting models. In *2019 8th
  Brazilian Conference on Intelligent Systems (BRACIS)* (pp. 872–877). IEEE.
  `doi:10.1109/BRACIS.2019.00155 <https://doi.org/10.1109/BRACIS.2019.00155>`__

  O resíduo do corretor ainda pode ter estrutura e ser previsto por outro modelo. *Na
  AutoHybTS:* ``residual_multistage``, até 3 estágios.

* **de Oliveira, J. F. L., Pacífico, L. D. S., de Mattos Neto, P. S. G., et al.
  (2020).** A hybrid optimized error correction system for time series forecasting.
  *Applied Soft Computing*, 87, 105970.
  `doi:10.1016/j.asoc.2019.105970 <https://doi.org/10.1016/j.asoc.2019.105970>`__

  *Contexto:* otimiza a forma de juntar a previsão da base e a correção. Na AutoHybTS, o
  peso da correção é o ``alpha`` (0,5 ou 1), escolhido na validação.

* **Smyl, S. (2020).** A hybrid method of exponential smoothing and recurrent neural
  networks for time series forecasting. *International Journal of Forecasting*, 36(1),
  75–85.
  `doi:10.1016/j.ijforecast.2019.03.017 <https://doi.org/10.1016/j.ijforecast.2019.03.017>`__

  *Contexto:* híbrido de suavização exponencial e LSTM treinada em muitas séries,
  vencedor do M4. A AutoHybTS ajusta modelos locais, série a série.

* **Santos Júnior, D. S. de O., de Mattos Neto, P. S. G., de Oliveira, J. F. L., &
  Cavalcanti, G. D. C. (2023).** A hybrid system based on ensemble learning to model
  residuals for time series forecasting. *Information Sciences*, 649, 119614.
  `doi:10.1016/j.ins.2023.119614 <https://doi.org/10.1016/j.ins.2023.119614>`__

  ERF: um ensemble de corretores no lugar de um corretor único. *Na AutoHybTS:*
  combinação de corretores (``arima-comb(mlp,knn)``), com corretores de tipos diferentes
  e pesos ``equal`` ou ``inverse_mse``; o ERF agrega corretores do mesmo tipo treinados
  em reamostras.

Decomposição
~~~~~~~~~~~~

* **Cleveland, R. B., Cleveland, W. S., McRae, J. E., & Terpenning, I. (1990).** STL: A
  seasonal-trend decomposition procedure based on loess. *Journal of Official
  Statistics*, 6(1), 3–73.
  `PDF (Statistics Sweden) <https://www.scb.se/contentassets/ca21efb41fee47d293bbee5bf7be7fb3/stl-a-seasonal-trend-decomposition-procedure-based-on-loess.pdf>`__

  *Na AutoHybTS:* ``stl(...)``, STL robusta ajustada só no treino; só na variante A.

* **Babu, C. N., & Reddy, B. E. (2014).** A moving-average filter based hybrid
  ARIMA–ANN model for forecasting time series data. *Applied Soft Computing*, 23, 27–38.
  `doi:10.1016/j.asoc.2014.05.028 <https://doi.org/10.1016/j.asoc.2014.05.028>`__

  Um filtro de média móvel separa a série em componentes de baixa e de alta
  volatilidade; o ARIMA prevê a primeira e uma rede neural, a segunda. O comprimento do
  filtro é escolhido pela curtose (alvo 3). *Na AutoHybTS:* ``ma(...)``, média móvel
  causal com janela pela curtose no treino.

* **de Oliveira, J. F. L., & Ludermir, T. B. (2016).** A hybrid evolutionary
  decomposition system for time series forecasting. *Neurocomputing*, 180, 27–34.
  `doi:10.1016/j.neucom.2015.07.113 <https://doi.org/10.1016/j.neucom.2015.07.113>`__

  *Contexto:* decomposição da série com modelos por componente configurados por
  otimização evolucionária.

Seleção dinâmica e combinação local
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Silva, E. G., Cavalcanti, G. D. C., de Oliveira, J. F. L., & de Mattos Neto, P. S.
  G. (2020).** On the evaluation of dynamic selection parameters for time series
  forecasting. In *2020 International Joint Conference on Neural Networks (IJCNN)*
  (pp. 1–7). IEEE.
  `doi:10.1109/IJCNN48605.2020.9207222 <https://doi.org/10.1109/IJCNN48605.2020.9207222>`__

  Efeito do tamanho da região de competência, da medida de similaridade e do grupo de
  modelos. *Na AutoHybTS:* referência da grade de ``k`` (vizinhos) e ``lag`` (janela) de
  ``dref`` e ``local`` na busca.

* **de Oliveira, J. F. L., Silva, E. G., & de Mattos Neto, P. S. G. (2022).** A hybrid
  system based on dynamic selection for time series forecasting. *IEEE Transactions on
  Neural Networks and Learning Systems*, 33(8), 3251–3263.
  `doi:10.1109/TNNLS.2021.3051384 <https://doi.org/10.1109/TNNLS.2021.3051384>`__

  DReF: a cada passo, escolhe o corretor do resíduo — ou nenhum — pelo desempenho nos
  vizinhos mais parecidos. *Na AutoHybTS:* ``dref(...)``, só na variante B
  (:doc:`user_guide/variant_b`).

* **Medina, M. C. C., & de Oliveira, J. F. L. (2025).** A local learning approach for
  sequential hybrid systems in time series forecasting. *Neurocomputing*, 620, 129235.
  `doi:10.1016/j.neucom.2024.129235 <https://doi.org/10.1016/j.neucom.2024.129235>`__

  Região de competência por vizinhos mais próximos e combinação local por Kernel Ridge.
  *Na AutoHybTS:* ``local``, só na variante B.

* **Carvalho, H. D. P. de, Oliveira, J. F. L. de, & Fagundes, R. A. de A. (2025).**
  Dynamic selection of ensemble-based regression models: Systematic literature review.
  *Expert Systems with Applications*, 290, 128429.
  `doi:10.1016/j.eswa.2025.128429 <https://doi.org/10.1016/j.eswa.2025.128429>`__

  *Contexto:* revisão sistemática de seleção e ponderação dinâmicas em regressão.

Modelos-base e corretores
-------------------------

* **Yule, G. U. (1927).** On a method of investigating periodicities in disturbed
  series, with special reference to Wolfer's sunspot numbers. *Philosophical
  Transactions of the Royal Society of London. Series A*, 226, 267–298.
  `doi:10.1098/rsta.1927.0007 <https://doi.org/10.1098/rsta.1927.0007>`__

  O modelo autorregressivo. *Na AutoHybTS:* regressões sobre defasagens (``linear`` e
  os demais modelos sobre lags) e ``ar_lags`` da ``harmonic_regression``.

* **Holt, C. C. (1957; republicado em 2004).** Forecasting seasonals and trends by
  exponentially weighted moving averages. *International Journal of Forecasting*,
  20(1), 5–10.
  `doi:10.1016/j.ijforecast.2003.09.015 <https://doi.org/10.1016/j.ijforecast.2003.09.015>`__

  Suavização exponencial com nível e tendência. *Na AutoHybTS:*
  ``exponential_smoothing`` (``trend``).

* **Winters, P. R. (1960).** Forecasting sales by exponentially weighted moving
  averages. *Management Science*, 6(3), 324–342.
  `doi:10.1287/mnsc.6.3.324 <https://doi.org/10.1287/mnsc.6.3.324>`__

  Holt-Winters, com sazonalidade. *Na AutoHybTS:* ``exponential_smoothing``
  (``season_length``).

* **Hoerl, A. E., & Kennard, R. W. (1970).** Ridge regression: Biased estimation for
  nonorthogonal problems. *Technometrics*, 12(1), 55–67.
  `doi:10.1080/00401706.1970.10488634 <https://doi.org/10.1080/00401706.1970.10488634>`__

  *Na AutoHybTS:* ``ridge_regression``.

* **Box, G. E. P., & Jenkins, G. M. (1970).** *Time series analysis: Forecasting and
  control*. Holden-Day. (Livro, sem DOI.)

  Identificação, estimação e diagnóstico de modelos ARIMA e ARIMA sazonais. *Na
  AutoHybTS:* ``arima`` e ``sarima``.

* **Akaike, H. (1974).** A new look at the statistical model identification. *IEEE
  Transactions on Automatic Control*, 19(6), 716–723.
  `doi:10.1109/TAC.1974.1100705 <https://doi.org/10.1109/TAC.1974.1100705>`__

  O AIC. *Na AutoHybTS:* escolha de ``p`` e ``q`` do ``arima``.

* **Breiman, L., Friedman, J. H., Olshen, R. A., & Stone, C. J. (1984).**
  *Classification and regression trees*. Wadsworth. Reedição CRC Press,
  `doi:10.1201/9781315139470 <https://doi.org/10.1201/9781315139470>`__

  *Na AutoHybTS:* ``tree``.

* **Gardner, E. S., & McKenzie, E. (1985).** Forecasting trends in time series.
  *Management Science*, 31(10), 1237–1246.
  `doi:10.1287/mnsc.31.10.1237 <https://doi.org/10.1287/mnsc.31.10.1237>`__

  Tendência amortecida. *Na AutoHybTS:* ``exponential_smoothing`` (``damped``).

* **Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986).** Learning
  representations by back-propagating errors. *Nature*, 323(6088), 533–536.
  `doi:10.1038/323533a0 <https://doi.org/10.1038/323533a0>`__

  *Na AutoHybTS:* ``mlp`` (base e corretor).

* **Kwiatkowski, D., Phillips, P. C. B., Schmidt, P., & Shin, Y. (1992).** Testing the
  null hypothesis of stationarity against the alternative of a unit root. *Journal of
  Econometrics*, 54(1–3), 159–178.
  `doi:10.1016/0304-4076(92)90104-Y <https://doi.org/10.1016/0304-4076(92)90104-Y>`__

  O teste KPSS. *Na AutoHybTS:* escolha de ``d`` do ``arima``.

* **Altman, N. S. (1992).** An introduction to kernel and nearest-neighbor
  nonparametric regression. *The American Statistician*, 46(3), 175–185.
  `doi:10.1080/00031305.1992.10475879 <https://doi.org/10.1080/00031305.1992.10475879>`__

  *Na AutoHybTS:* ``knn`` (base e corretor).

* **Tibshirani, R. (1996).** Regression shrinkage and selection via the lasso. *Journal
  of the Royal Statistical Society: Series B*, 58(1), 267–288.
  `doi:10.1111/j.2517-6161.1996.tb02080.x <https://doi.org/10.1111/j.2517-6161.1996.tb02080.x>`__

  *Na AutoHybTS:* ``lasso_regression``.

* **Drucker, H., Burges, C. J. C., Kaufman, L., Smola, A., & Vapnik, V. (1997).**
  Support vector regression machines. In *Advances in Neural Information Processing
  Systems 9* (NIPS 1996). MIT Press.
  `NeurIPS Proceedings <https://proceedings.neurips.cc/paper/1996/hash/d38901788c533e8286cb6400b40b386d-Abstract.html>`__

  *Na AutoHybTS:* ``svm``.

* **Drucker, H. (1997).** Improving regressors using boosting techniques. In
  *Proceedings of the 14th International Conference on Machine Learning (ICML 1997)*
  (pp. 107–115). Morgan Kaufmann. (Sem DOI.)

  AdaBoost.R2. *Na AutoHybTS:* ``adaboost_r2``.

* **Hochreiter, S., & Schmidhuber, J. (1997).** Long short-term memory. *Neural
  Computation*, 9(8), 1735–1780.
  `doi:10.1162/neco.1997.9.8.1735 <https://doi.org/10.1162/neco.1997.9.8.1735>`__

  *Na AutoHybTS:* ``LSTMNet`` (PyTorch) e ``lstm`` (Keras), exemplos de modelos
  registrados (:doc:`user_guide/custom_models`).

* **Breiman, L. (2001).** Random forests. *Machine Learning*, 45(1), 5–32.
  `doi:10.1023/A:1010933404324 <https://doi.org/10.1023/A:1010933404324>`__

  *Na AutoHybTS:* ``random_forest``.

* **Friedman, J. H. (2001).** Greedy function approximation: A gradient boosting
  machine. *The Annals of Statistics*, 29(5), 1189–1232.
  `doi:10.1214/aos/1013203451 <https://doi.org/10.1214/aos/1013203451>`__

  *Na AutoHybTS:* ``hist_gradient_boosting`` e o princípio de ``xgboost_regressor``,
  ``lightgbm_regressor`` e ``catboost_regressor``.

* **Smola, A. J., & Schölkopf, B. (2004).** A tutorial on support vector regression.
  *Statistics and Computing*, 14(3), 199–222.
  `doi:10.1023/B:STCO.0000035301.49549.88 <https://doi.org/10.1023/B:STCO.0000035301.49549.88>`__

  *Na AutoHybTS:* ``svm``.

* **Zou, H., & Hastie, T. (2005).** Regularization and variable selection via the
  elastic net. *Journal of the Royal Statistical Society: Series B*, 67(2), 301–320.
  `doi:10.1111/j.1467-9868.2005.00503.x <https://doi.org/10.1111/j.1467-9868.2005.00503.x>`__

  *Na AutoHybTS:* ``elastic_net_regression``.

* **Geurts, P., Ernst, D., & Wehenkel, L. (2006).** Extremely randomized trees.
  *Machine Learning*, 63(1), 3–42.
  `doi:10.1007/s10994-006-6226-1 <https://doi.org/10.1007/s10994-006-6226-1>`__

  *Na AutoHybTS:* ``extra_trees_regressor``.

* **Hyndman, R. J., & Khandakar, Y. (2008).** Automatic time series forecasting: The
  forecast package for R. *Journal of Statistical Software*, 27(3).
  `doi:10.18637/jss.v027.i03 <https://doi.org/10.18637/jss.v027.i03>`__

  ``auto.arima``: ``d`` por testes KPSS sucessivos e ``p``, ``q`` por critério de
  informação. *Na AutoHybTS:* ``arima`` com ``order="auto"`` segue a mesma ideia (KPSS e
  AIC, só no treino); ``sarima`` acrescenta ``D`` pela força sazonal e a busca *stepwise*
  de ``(p, q, P, Q)`` por AICc a partir de quatro modelos iniciais.

* **Chen, T., & Guestrin, C. (2016).** XGBoost: A scalable tree boosting system. In
  *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery
  and Data Mining* (pp. 785–794).
  `doi:10.1145/2939672.2939785 <https://doi.org/10.1145/2939672.2939785>`__

  *Na AutoHybTS:* ``xgboost_regressor``.

* **Ke, G., Meng, Q., Finley, T., et al. (2017).** LightGBM: A highly efficient gradient
  boosting decision tree. In *Advances in Neural Information Processing Systems 30*.
  `NeurIPS Proceedings <https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html>`__

  *Na AutoHybTS:* ``lightgbm_regressor``; o ``HistGradientBoostingRegressor`` do
  scikit-learn se inspira nele.

* **Vaswani, A., Shazeer, N., Parmar, N., et al. (2017).** Attention is all you need.
  In *Advances in Neural Information Processing Systems 30*.
  `NeurIPS Proceedings <https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html>`__

  *Na AutoHybTS:* ``TransformerNet`` (PyTorch) e ``transformer`` (Keras), exemplos de
  modelos registrados.

* **Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018).**
  CatBoost: Unbiased boosting with categorical features. In *Advances in Neural
  Information Processing Systems 31*.
  `NeurIPS Proceedings <https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html>`__

  *Na AutoHybTS:* ``catboost_regressor``.

* **Hyndman, R. J., & Athanasopoulos, G. (2021).** *Forecasting: Principles and
  practice* (3ª ed.). OTexts. `otexts.com/fpp3 <https://otexts.com/fpp3/>`__

  Métodos simples (`seção 5.2 <https://otexts.com/fpp3/toolbox.html#sec-simple-methods>`__)
  e regressão harmônica dinâmica
  (`seção 10.5 <https://otexts.com/fpp3/dynamic-regression.html#sec-dhr>`__). *Na
  AutoHybTS:* ``naive``, ``seasonal_naive``, ``random_walk_drift``,
  ``historic_average`` e ``harmonic_regression``.

* **Gu, A., & Dao, T. (2023).** Mamba: Linear-time sequence modeling with selective
  state spaces. arXiv:2312.00752.
  `arXiv <https://arxiv.org/abs/2312.00752>`__

  *Na AutoHybTS:* ``MambaNet``, versão compacta em PyTorch puro (não o pacote
  ``mamba-ssm``).

Demanda intermitente
--------------------

* **Croston, J. D. (1972).** Forecasting and stock control for intermittent demands.
  *Operational Research Quarterly* (hoje *Journal of the Operational Research
  Society*), 23(3), 289–303.
  `doi:10.1057/jors.1972.50 <https://doi.org/10.1057/jors.1972.50>`__

  Suaviza separadamente o tamanho das demandas e o intervalo entre elas. *Na
  AutoHybTS:* base do ``croston_sba`` (:doc:`user_guide/models`).

* **Syntetos, A. A., & Boylan, J. E. (2005).** The accuracy of intermittent demand
  estimates. *International Journal of Forecasting*, 21(2), 303–314.
  `doi:10.1016/j.ijforecast.2004.10.001 <https://doi.org/10.1016/j.ijforecast.2004.10.001>`__

  Correção do viés de Croston, ``(1 − α/2) · z / p``. *Na AutoHybTS:* ``croston_sba``.

* **Syntetos, A. A., Boylan, J. E., & Croston, J. D. (2005).** On the categorization of
  demand patterns. *Journal of the Operational Research Society*, 56(5), 495–503.
  `doi:10.1057/palgrave.jors.2601841 <https://doi.org/10.1057/palgrave.jors.2601841>`__

  Cortes ADI = 1,32 e CV² = 0,49 entre demanda suave, errática, intermitente e
  irregular. *Na AutoHybTS:* perfil da série nos diagnósticos (fração de zeros, ADI,
  CV²).

* **Nikolopoulos, K., Syntetos, A. A., Boylan, J. E., Petropoulos, F., &
  Assimakopoulos, V. (2011).** An aggregate–disaggregate intermittent demand approach
  (ADIDA) to forecasting: An empirical proposition and analysis. *Journal of the
  Operational Research Society*, 62(3), 544–554.
  `doi:10.1057/jors.2010.32 <https://doi.org/10.1057/jors.2010.32>`__

  *Na AutoHybTS:* ``adida``.

* **Teunter, R. H., Syntetos, A. A., & Babai, M. Z. (2011).** Intermittent demand:
  Linking forecasting to inventory obsolescence. *European Journal of Operational
  Research*, 214(3), 606–615.
  `doi:10.1016/j.ejor.2011.05.018 <https://doi.org/10.1016/j.ejor.2011.05.018>`__

  *Na AutoHybTS:* ``tsb``.

* **Petropoulos, F., & Kourentzes, N. (2015).** Forecast combinations for intermittent
  demand. *Journal of the Operational Research Society*, 66(6), 914–924.
  `doi:10.1057/jors.2014.62 <https://doi.org/10.1057/jors.2014.62>`__

  IMAPA: média das previsões em vários níveis de agregação. *Na AutoHybTS:* ``imapa``.

Avaliação, validação temporal e testes
--------------------------------------

* **Friedman, M. (1937).** The use of ranks to avoid the assumption of normality
  implicit in the analysis of variance. *Journal of the American Statistical
  Association*, 32(200), 675–701.
  `doi:10.1080/01621459.1937.10503522 <https://doi.org/10.1080/01621459.1937.10503522>`__

  *Na AutoHybTS:* ``cmp.friedman()`` (:doc:`user_guide/comparison`) e o teste de
  sazonalidade de ``seasonality_report`` e ``seasonal_period="auto"``, com os ciclos
  como blocos (:doc:`user_guide/diagnostics`).

* **Wilcoxon, F. (1945).** Individual comparisons by ranking methods. *Biometrics
  Bulletin*, 1(6), 80–83.
  `doi:10.2307/3001968 <https://doi.org/10.2307/3001968>`__

  *Na AutoHybTS:* ``cmp.tests()`` e a opção de teste pareado unilateral da regra de
  aceite (:doc:`user_guide/selection`).

* **Holm, S. (1979).** A simple sequentially rejective multiple test procedure.
  *Scandinavian Journal of Statistics*, 6(2), 65–70.
  `JSTOR <https://www.jstor.org/stable/4615733>`__

  *Na AutoHybTS:* ajuste dos p-valores em ``cmp.tests()``.

* **Efron, B. (1979).** Bootstrap methods: Another look at the jackknife. *The Annals of
  Statistics*, 7(1), 1–26.
  `doi:10.1214/aos/1176344552 <https://doi.org/10.1214/aos/1176344552>`__

  *Na AutoHybTS:* intervalos da calibração, reamostrando séries
  (:doc:`user_guide/benchmarks`).

* **Newey, W. K., & West, K. D. (1987).** A simple, positive semi-definite,
  heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*,
  55(3), 703–708.
  `doi:10.2307/1913610 <https://doi.org/10.2307/1913610>`__

  *Na AutoHybTS:* variância de longo prazo no teste de Diebold-Mariano, para origens que
  se sobrepõem.

* **Makridakis, S. (1993).** Accuracy measures: Theoretical and practical concerns.
  *International Journal of Forecasting*, 9(4), 527–529.
  `doi:10.1016/0169-2070(93)90079-3 <https://doi.org/10.1016/0169-2070(93)90079-3>`__

  *Na AutoHybTS:* métrica ``sMAPE``.

* **Diebold, F. X., & Mariano, R. S. (1995).** Comparing predictive accuracy. *Journal
  of Business & Economic Statistics*, 13(3), 253–263.
  `doi:10.1080/07350015.1995.10524599 <https://doi.org/10.1080/07350015.1995.10524599>`__

  *Na AutoHybTS:* ``cmp.tests()``.

* **Tashman, L. J. (2000).** Out-of-sample tests of forecasting accuracy: An analysis
  and review. *International Journal of Forecasting*, 16(4), 437–450.
  `doi:10.1016/S0169-2070(00)00065-0 <https://doi.org/10.1016/S0169-2070(00)00065-0>`__

  Avaliação com origem rolante. *Na AutoHybTS:* origens de ``V_search`` e ``V_select`` e
  ``cross_validate`` (:doc:`user_guide/temporal_validation`).

* **Demšar, J. (2006).** Statistical comparisons of classifiers over multiple data sets.
  *Journal of Machine Learning Research*, 7, 1–30.
  `JMLR <https://jmlr.org/papers/v7/demsar06a.html>`__

  Teste de Friedman seguido da diferença crítica de Nemenyi. *Na AutoHybTS:*
  ``cmp.friedman()`` e seu gráfico.

* **Hyndman, R. J., & Koehler, A. B. (2006).** Another look at measures of forecast
  accuracy. *International Journal of Forecasting*, 22(4), 679–688.
  `doi:10.1016/j.ijforecast.2006.03.001 <https://doi.org/10.1016/j.ijforecast.2006.03.001>`__

  *Na AutoHybTS:* ``MASE``, a métrica padrão de ordenação e seleção.

* **Wang, X., Smith, K., & Hyndman, R. (2006).** Characteristic-based clustering for
  time series data. *Data Mining and Knowledge Discovery*, 13(3), 335–364.
  `doi:10.1007/s10618-005-0039-x <https://doi.org/10.1007/s10618-005-0039-x>`__

  Força da tendência e da sazonalidade a partir de uma decomposição. *Na AutoHybTS:*
  ``strength`` de ``seasonality_report`` e a escolha de ``D`` do ``sarima``.

* **Varma, S., & Simon, R. (2006).** Bias in error estimation when using cross-validation
  for model selection. *BMC Bioinformatics*, 7, 91.
  `doi:10.1186/1471-2105-7-91 <https://doi.org/10.1186/1471-2105-7-91>`__

  Escolher e avaliar nos mesmos dados subestima o erro; a validação cruzada aninhada
  corrige. *Na AutoHybTS:* ``nested_cross_validate``.

* **Cawley, G. C., & Talbot, N. L. C. (2010).** On over-fitting in model selection and
  subsequent selection bias in performance evaluation. *Journal of Machine Learning
  Research*, 11, 2079–2107.
  `JMLR <https://jmlr.org/papers/v11/cawley10a.html>`__

  *Na AutoHybTS:* orçamentos da busca, separação entre ``V_search`` e ``V_select`` e o
  teste selado.

* **Bergmeir, C., & Benítez, J. M. (2012).** On the use of cross-validation for time
  series predictor evaluation. *Information Sciences*, 191, 192–213.
  `doi:10.1016/j.ins.2011.12.028 <https://doi.org/10.1016/j.ins.2011.12.028>`__

  *Na AutoHybTS:* ``cross_validate``, com origens só no histórico de desenvolvimento.

* **Bergmeir, C., Hyndman, R. J., & Koo, B. (2018).** A note on the validity of
  cross-validation for evaluating autoregressive time series prediction.
  *Computational Statistics & Data Analysis*, 120, 70–83.
  `doi:10.1016/j.csda.2017.11.003 <https://doi.org/10.1016/j.csda.2017.11.003>`__

  *Contexto:* quando a validação cruzada comum vale para modelos autorregressivos. A
  AutoHybTS usa só origens temporais.

Competições e dados
-------------------

* **Makridakis, S., Andersen, A., Carbone, R., et al. (1982).** The accuracy of
  extrapolation (time series) methods: Results of a forecasting competition. *Journal
  of Forecasting*, 1(2), 111–153.
  `doi:10.1002/for.3980010202 <https://doi.org/10.1002/for.3980010202>`__

  *Contexto:* a primeira M-Competition.

* **Makridakis, S., & Hibon, M. (2000).** The M3-Competition: Results, conclusions and
  implications. *International Journal of Forecasting*, 16(4), 451–476.
  `doi:10.1016/S0169-2070(00)00057-1 <https://doi.org/10.1016/S0169-2070(00)00057-1>`__

  *Na AutoHybTS:* séries do M3 em ``get_data`` (:doc:`user_guide/datasets`).

* **Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2020).** The M4 Competition:
  100,000 time series and 61 forecasting methods. *International Journal of
  Forecasting*, 36(1), 54–74.
  `doi:10.1016/j.ijforecast.2019.04.014 <https://doi.org/10.1016/j.ijforecast.2019.04.014>`__

  *Na AutoHybTS:* séries do M4 em ``get_data``; dados do repositório
  `M4-methods <https://github.com/Mcompetitions/M4-methods>`__.

* **Montero-Manso, P., & Hyndman, R. J. (2021).** Principles and algorithms for
  forecasting groups of time series: Locality and globality. *International Journal of
  Forecasting*, 37(4), 1632–1653.
  `doi:10.1016/j.ijforecast.2021.03.004 <https://doi.org/10.1016/j.ijforecast.2021.03.004>`__

  *Contexto:* modelos locais e globais. Nos painéis, a AutoHybTS ajusta um modelo por
  série (:doc:`user_guide/panels`).

* **Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2022).** M5 accuracy
  competition: Results, findings, and conclusions. *International Journal of
  Forecasting*, 38(4), 1346–1364.
  `doi:10.1016/j.ijforecast.2021.11.013 <https://doi.org/10.1016/j.ijforecast.2021.11.013>`__

  *Na AutoHybTS:* métrica ``RMSSE``.

* **Arel-Bundock, V. (s.d.).** Rdatasets: An archive of datasets distributed with R.
  `vincentarelbundock.github.io/Rdatasets <https://vincentarelbundock.github.io/Rdatasets/>`__

  *Na AutoHybTS:* séries do Rdatasets em ``get_data``.

Software
--------

* **Seabold, S., & Perktold, J. (2010).** Statsmodels: Econometric and statistical
  modeling with Python. In *Proceedings of the 9th Python in Science Conference*
  (pp. 92–96).
  `doi:10.25080/Majora-92bf1922-011 <https://doi.org/10.25080/Majora-92bf1922-011>`__

  *Na AutoHybTS:* ``arima``, ``exponential_smoothing``, STL e KPSS (extra ``stats``).

* **Bergstra, J., Bardenet, R., Bengio, Y., & Kégl, B. (2011).** Algorithms for
  hyper-parameter optimization. In *Advances in Neural Information Processing Systems
  24*.
  `NeurIPS Proceedings <https://proceedings.neurips.cc/paper/2011/hash/86e8f7ab32cfd12577bc2619bc635690-Abstract.html>`__

  O TPE. *Na AutoHybTS:* ``tune_model(sampler="optuna")`` (:doc:`user_guide/tuning`).

* **Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011).** Scikit-learn: Machine
  learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
  `JMLR <https://jmlr.org/papers/v12/pedregosa11a.html>`__

  *Na AutoHybTS:* modelos sobre lags e corretores.

* **Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019).** Optuna: A
  next-generation hyperparameter optimization framework. In *Proceedings of the 25th
  ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*
  (pp. 2623–2631).
  `doi:10.1145/3292500.3330701 <https://doi.org/10.1145/3292500.3330701>`__

  *Na AutoHybTS:* ``tune_model(sampler="optuna")``.
