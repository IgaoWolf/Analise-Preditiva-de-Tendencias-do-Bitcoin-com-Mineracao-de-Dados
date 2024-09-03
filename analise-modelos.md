Análise dos Resultados
Regressão Logística:

Acurácia: 65.84%
Matriz de Confusão: Mostra que o modelo teve dificuldades tanto para prever "False" quanto "True".
F1-Score: Médio, indicando que o modelo tem um desempenho moderado em previsões binárias.
Árvore de Decisão:

Acurácia: 57.30%
Matriz de Confusão: O modelo teve dificuldades para prever corretamente ambas as classes, apresentando a pior acurácia entre os modelos testados.
F1-Score: Mais baixo, indicando desempenho inferior.
Floresta Aleatória:

Acurácia: 63.91%
Matriz de Confusão: Desempenho moderado, um pouco melhor do que a Árvore de Decisão, mas ainda não satisfatório.
F1-Score: Melhor que a Árvore de Decisão, mas ainda baixo.
SVC (Support Vector Classifier):

Acurácia: 65.56%
Matriz de Confusão: Bom recall para a classe "False", mas desempenho inferior para a classe "True".
F1-Score: Indica que o modelo tem um equilíbrio melhor entre precisão e recall, mas ainda abaixo do ideal.
MLPClassifier (Rede Neural Perceptron Multi-Camadas):

Acurácia: 89.53%
Matriz de Confusão: Ótimo desempenho, com precisão e recall elevados para ambas as classes.
F1-Score: Alto para ambas as classes, indicando que o modelo é bastante eficaz em prever se o preço do BTC sobe ou desce.