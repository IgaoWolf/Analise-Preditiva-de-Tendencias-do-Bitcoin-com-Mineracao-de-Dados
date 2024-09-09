### Importações

```python
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os
```

- **`pandas`** (`pd`): Biblioteca para manipulação de dados, especialmente de tabelas de dados.
- **`sklearn.model_selection.GridSearchCV`**: Função que realiza a busca de hiperparâmetros em grade para otimizar o desempenho de um modelo de aprendizado de máquina.
- **`sklearn.neural_network.MLPClassifier`**: Um classificador de rede neural feedforward de múltiplas camadas (MLP).
- **`sklearn.metrics`**: Ferramentas para medir o desempenho do modelo, como `classification_report`, `confusion_matrix`, e `accuracy_score`.
- **`os`**: Biblioteca padrão do Python para interagir com o sistema operacional, usada aqui para manipular caminhos de diretórios.

### Definição de Diretórios

```python
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
train_file = os.path.join(base_dir, 'data/processed/train_data.csv')
test_file = os.path.join(base_dir, 'data/processed/test_data.csv')
```

- **`base_dir`**: Define o diretório base do projeto usando o caminho do arquivo atual (`__file__`) e subindo dois níveis na hierarquia de diretórios (`../../`).
- **`train_file`** e **`test_file`**: Definem os caminhos completos para os arquivos CSV dos dados de treino e teste que serão utilizados para o modelo de machine learning.

### Função para Carregar Dados

```python
def load_data(train_path, test_path):
    """
    Carrega os conjuntos de dados de treino e teste.

    Parâmetros:
    - train_path (str): Caminho para o arquivo de treino.
    - test_path (str): Caminho para o arquivo de teste.

    Retorno:
    - Dois DataFrames do pandas: dados de treino e dados de teste.
    """
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    return train_data, test_data
```

- **Função `load_data`**:
  - Carrega os conjuntos de dados de treino e teste a partir dos caminhos especificados.
  - Retorna dois DataFrames do `pandas`: um para os dados de treino e outro para os dados de teste.

### Função para Treinar e Avaliar o Modelo

```python
def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Treina e avalia um modelo de aprendizado de máquina.

    Parâmetros:
    - model: Algoritmo de aprendizado de máquina a ser treinado.
    - X_train: Dados de entrada de treino.
    - y_train: Rótulos de treino.
    - X_test: Dados de entrada de teste.
    - y_test: Rótulos de teste.

    Retorno:
    - Exibe a acurácia, a matriz de confusão e o relatório de classificação do modelo.
    """
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    print(f"\nModelo: {model.__class__.__name__} (Otimizado)")
    print("Acurácia:", accuracy_score(y_test, predictions))
    print("Matriz de Confusão:\n", confusion_matrix(y_test, predictions))
    print("Relatório de Classificação:\n", classification_report(y_test, predictions))
```

- **Função `train_and_evaluate_model`**:
  - Treina o modelo de machine learning usando os dados de treino (`X_train` e `y_train`).
  - Faz previsões nos dados de teste (`X_test`) e compara as previsões com os rótulos verdadeiros (`y_test`).
  - Exibe a acurácia, matriz de confusão, e o relatório de classificação do modelo treinado.

### Execução Principal

```python
if __name__ == "__main__":
    # Carrega os dados de treino e teste
    train_data, test_data = load_data(train_file, test_file)

    # Separar as features (X) e o target (y)
    X_train = train_data.drop(['Date', 'Close'], axis=1)  # Remove as colunas 'Date' e 'Close'
    y_train = train_data['Close'] > train_data['Close'].shift(1)  # Define o target como subida ou descida
    X_test = test_data.drop(['Date', 'Close'], axis=1)
    y_test = test_data['Close'] > test_data['Close'].shift(1)

    # Ajuste dos parâmetros para o Grid Search com mais iterações
    param_grid = {
        'hidden_layer_sizes': [(100,), (50, 50), (100, 50)],  # Testa diferentes tamanhos de camadas ocultas
        'activation': ['tanh', 'relu'],  # Testa diferentes funções de ativação
        'solver': ['adam'],  # Usa o otimizador 'adam'
        'alpha': [0.0001, 0.001],  # Regularização
        'learning_rate': ['constant', 'adaptive'],  # Diferentes métodos de taxa de aprendizado
        'max_iter': [2000]  # Aumenta o número de iterações para melhorar a convergência
    }

    # Configurando o Grid Search
    grid_search = GridSearchCV(MLPClassifier(), param_grid, n_jobs=-1, cv=3)

    # Executando o Grid Search para encontrar os melhores parâmetros
    print("Iniciando o Grid Search para otimizar o MLPClassifier com mais iterações...")
    grid_search.fit(X_train, y_train)

    # Exibindo os melhores parâmetros encontrados
    print("Melhores parâmetros encontrados:", grid_search.best_params_)

    # Avaliando o melhor modelo encontrado no Grid Search
    best_model = grid_search.best_estimator_
    train_and_evaluate_model(best_model, X_train, y_train, X_test, y_test)
```

- **Separação dos Dados**:
  - **`X_train`** e **`X_test`**: Variáveis independentes (features) obtidas removendo as colunas `'Date'` e `'Close'` dos dados de treino e teste.
  - **`y_train`** e **`y_test`**: Variáveis dependentes (target), definidas como um valor booleano indicando se o preço de fechamento aumentou ou diminuiu em relação ao dia anterior (`Close.shift(1)`).

- **Definição da Grade de Parâmetros (`param_grid`)**:
  - Define diferentes configurações para os hiperparâmetros do `MLPClassifier`:
    - **`hidden_layer_sizes`**: Testa diferentes tamanhos de camadas ocultas.
    - **`activation`**: Testa diferentes funções de ativação, como `'tanh'` e `'relu'`.
    - **`solver`**: Usa o otimizador `'adam'`.
    - **`alpha`**: Define o coeficiente de regularização.
    - **`learning_rate`**: Testa diferentes métodos de taxa de aprendizado.
    - **`max_iter`**: Define o número máximo de iterações para o algoritmo de treinamento.

- **Configuração e Execução do Grid Search**:
  - **`GridSearchCV`**: Executa a busca em grade para encontrar a melhor combinação de hiperparâmetros para o `MLPClassifier`.
  - Treina o modelo com a melhor combinação de hiperparâmetros encontrados.

- **Avaliação do Melhor Modelo**:
  - Avalia o melhor modelo encontrado pela busca em grade usando a função `train_and_evaluate_model`.

### Resumo

Este script realiza um treinamento e otimização de um modelo de rede neural (`MLPClassifier`) para prever a direção do preço do Bitcoin (subida ou descida). O processo inclui:

1. **Carregamento dos Dados**: Carrega os dados de treino e teste a partir de arquivos CSV.
2. **Separação das Variáveis**: Separa as variáveis de entrada (features) e de saída (target).
3. **Otimização dos Hiperparâmetros**: Utiliza `GridSearchCV` para encontrar a melhor combinação de hiperparâmetros.
4. **Treinamento e Avaliação do Modelo**: Treina o modelo com os melhores hiperparâmetros e avalia seu desempenho usando métricas como acurácia, matriz de confusão e relatório de classificação.

A modularidade do código facilita a adaptação para outros conjuntos de dados ou modelos, permitindo ajustes nos parâmetros conforme necessário.