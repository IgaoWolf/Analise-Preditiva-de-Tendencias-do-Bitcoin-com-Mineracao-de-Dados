### Importações

```python
import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os
import joblib  # Importa joblib para salvar o modelo
```

- **`pandas`** (`pd`): Biblioteca para manipulação e análise de dados estruturados.
- **`GridSearchCV`, `cross_val_score`, `KFold`**: Ferramentas do `sklearn` para otimização e validação de modelos.
  - **`GridSearchCV`**: Busca em grade para encontrar os melhores hiperparâmetros (não é usado diretamente neste código).
  - **`cross_val_score`**: Realiza a validação cruzada do modelo.
  - **`KFold`**: Divide os dados em partes para validação cruzada.
- **`MLPClassifier`**: Algoritmo de rede neural perceptron de múltiplas camadas.
- **`classification_report`, `confusion_matrix`, `accuracy_score`**: Funções de avaliação para medir o desempenho dos modelos.
- **`os`**: Biblioteca padrão do Python para interagir com o sistema operacional, usada aqui para manipular caminhos de diretórios.
- **`joblib`**: Biblioteca para salvar e carregar modelos treinados.

### Definição de Diretórios

```python
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
train_file = os.path.join(base_dir, 'data/processed/train_data.csv')
test_file = os.path.join(base_dir, 'data/processed/test_data.csv')
model_dir = os.path.join(base_dir, 'models')
model_save_path = os.path.join(model_dir, 'best_model.pkl')
```

- **`base_dir`**: Define o diretório base do projeto usando o caminho do arquivo atual (`__file__`) e subindo dois níveis na hierarquia de diretórios (`../../`).
- **`train_file`** e **`test_file`**: Caminhos completos para os arquivos CSV dos dados de treino e teste que serão utilizados para o modelo de aprendizado de máquina.
- **`model_dir`**: Diretório onde o modelo otimizado será salvo.
- **`model_save_path`**: Caminho completo para salvar o modelo otimizado em formato `.pkl`.

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
  - Treina o modelo de aprendizado de máquina usando os dados de treino (`X_train` e `y_train`).
  - Faz previsões nos dados de teste (`X_test`) e compara as previsões com os rótulos verdadeiros (`y_test`).
  - Exibe a acurácia, matriz de confusão e o relatório de classificação do modelo treinado.

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

    # Melhor modelo encontrado anteriormente
    best_model = MLPClassifier(activation='tanh', alpha=0.0001, hidden_layer_sizes=(50, 50), 
                               learning_rate='constant', max_iter=2000, solver='adam')

    # Validando o modelo com K-Fold Cross-Validation
    print("Iniciando a Validação Cruzada K-Fold...")
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_val_score(best_model, X_train, y_train, cv=kfold, scoring='accuracy')
    print(f"Acurácia média da Validação Cruzada: {cv_results.mean():.4f}")
    print(f"Desvio padrão da Acurácia: {cv_results.std():.4f}")

    # Treinando e avaliando o modelo no conjunto de teste
    train_and_evaluate_model(best_model, X_train, y_train, X_test, y_test)

    # Verificar se o diretório para salvar o modelo existe, e criar se não existir
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"Diretório criado: {model_dir}")

    # Salvar o modelo otimizado para uso em produção
    joblib.dump(best_model, model_save_path)
    print(f"Modelo otimizado salvo em: {model_save_path}")
```

- **Carregamento dos Dados**:
  - Utiliza a função `load_data` para carregar os dados de treino e teste a partir dos arquivos CSV.

- **Separação das Features (X) e Target (y)**:
  - **`X_train`** e **`X_test`**: Variáveis independentes (features), obtidas removendo as colunas `'Date'` e `'Close'` dos dados de treino e teste.
  - **`y_train`** e **`y_test`**: Variáveis dependentes (target), definidas como um valor booleano indicando se o preço de fechamento aumentou ou diminuiu em relação ao dia anterior (`Close.shift(1)`).

- **Modelo Otimizado (`best_model`)**:
  - Define o melhor modelo encontrado anteriormente usando `MLPClassifier` com parâmetros específicos, como `activation='tanh'`, `hidden_layer_sizes=(50, 50)`, e `max_iter=2000`.

- **Validação Cruzada K-Fold**:
  - Utiliza o K-Fold Cross-Validation com 5 divisões (splits) para validar o desempenho do modelo.
  - Exibe a acurácia média e o desvio padrão da acurácia obtidos durante a validação cruzada.

- **Treinamento e Avaliação do Modelo**:
  - Treina o modelo otimizado com o conjunto de dados de treino e avalia seu desempenho com o conjunto de dados de teste usando a função `train_and_evaluate_model`.

- **Salvar o Modelo Otimizado**:
  - Verifica se o diretório para salvar o modelo existe. Se não existir, cria o diretório.
  - Salva o modelo otimizado em formato `.pkl` usando `joblib.dump` para uso em produção.

### Resumo

Este script otimiza, valida e salva um modelo de rede neural para prever a direção do preço do Bitcoin:

1. **Carregamento dos Dados**: Carrega os dados de treino e teste a partir de arquivos CSV.
2. **Definição do Melhor Modelo**: Define um modelo de rede neural (`MLPClassifier`) com hiperparâmetros otimizados.
3. **Validação Cruzada K-Fold**: Avalia o modelo usando validação cruzada para garantir a robustez do modelo.
4. **Treinamento e Avaliação**: Treina o modelo otimizado com os dados de treino e avalia seu desempenho com os dados de teste.
5. **Salvar o Modelo

 Otimizado**: Salva o modelo treinado para uso futuro em produção.

A abordagem garante que o modelo seja bem avaliado e pronto para implementação prática.