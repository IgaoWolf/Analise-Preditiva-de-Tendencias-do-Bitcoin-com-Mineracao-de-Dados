### Importações

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
```

- **`pandas`** (`pd`): Biblioteca para manipulação e análise de dados estruturados.
- **`numpy`** (`np`): Biblioteca para operações matemáticas e manipulação de arrays.
- **`sklearn.preprocessing.MinMaxScaler`**: Ferramenta para normalização dos dados, escalando todos os valores para um intervalo entre 0 e 1.
- **`os`**: Biblioteca padrão do Python para interagir com o sistema operacional, usada aqui para manipular caminhos de diretórios.

### Definição de Diretórios

```python
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
input_file = os.path.join(base_dir, 'data/processed/clean_btc_data.csv')
output_file_train = os.path.join(base_dir, 'data/processed/train_data.csv')
output_file_test = os.path.join(base_dir, 'data/processed/test_data.csv')
```

- **`base_dir`**: Define o diretório base do projeto usando o caminho do arquivo atual (`__file__`) e subindo dois níveis na hierarquia de diretórios (`../../`).
- **`input_file`**: Caminho completo para o arquivo CSV contendo os dados limpos do Bitcoin.
- **`output_file_train`** e **`output_file_test`**: Caminhos completos para os arquivos CSV que conterão os dados de treino e teste, respectivamente.

### Função para Criar Novas Features

```python
def create_features(df):
    """
    Cria novos recursos (features) para o conjunto de dados.

    Parâmetros:
    - df (DataFrame): Conjunto de dados original.

    Retorno:
    - DataFrame com novos recursos.
    """
    # Calcular Médias Móveis Simples (SMA)
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    
    # Calcular Média Móvel Exponencial (EMA)
    df['EMA_5'] = df['Close'].ewm(span=5, adjust=False).mean()
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    
    # Calcular Índice de Força Relativa (RSI)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Calcular Momentum
    df['Momentum'] = df['Close'] - df['Close'].shift(4)

    # Remover linhas com valores nulos após a criação de novas features
    df = df.dropna()
    
    return df
```

- **Função `create_features`**:
  - Cria novos recursos a partir dos dados existentes para melhorar o desempenho dos algoritmos de aprendizado:
    - **Médias Móveis Simples (SMA)**: Calcula a média do preço de fechamento para as últimas 5 e 10 observações.
    - **Médias Móveis Exponenciais (EMA)**: Calcula a média exponencial ponderada do preço de fechamento para as últimas 5 e 10 observações.
    - **Índice de Força Relativa (RSI)**: Calcula o RSI, que é um indicador de momentum que mede a velocidade e a mudança dos movimentos de preços.
    - **Momentum**: Calcula a mudança no preço de fechamento em um período de 4 dias.
  - Remove quaisquer linhas que contenham valores nulos resultantes do cálculo de novas features.

### Função para Normalizar os Dados

```python
def normalize_data(df):
    """
    Normaliza os dados para melhorar a performance dos algoritmos de aprendizado.

    Parâmetros:
    - df (DataFrame): Conjunto de dados original.

    Retorno:
    - DataFrame normalizado.
    """
    # Remove a coluna de data antes da normalização
    df_features = df.drop(['Date'], axis=1)

    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_features), columns=df_features.columns)
    
    # Adiciona de volta a coluna 'Date' após a normalização
    df_scaled['Date'] = df['Date'].values
    
    return df_scaled
```

- **Função `normalize_data`**:
  - Normaliza todas as features para um intervalo entre 0 e 1 usando `MinMaxScaler`.
  - A normalização melhora a performance dos algoritmos de aprendizado de máquina, especialmente aqueles que são sensíveis às escalas de dados.
  - Remove a coluna de data antes de normalizar, pois datas não devem ser normalizadas, e adiciona-a novamente após a normalização.

### Função para Dividir os Dados

```python
def split_data(df, train_size=0.8):
    """
    Divide o conjunto de dados em conjuntos de treinamento e teste.

    Parâmetros:
    - df (DataFrame): Conjunto de dados.
    - train_size (float): Proporção dos dados para treinamento.

    Retorno:
    - Dois DataFrames: um para treino e outro para teste.
    """
    train_data = df[:int(len(df) * train_size)]
    test_data = df[int(len(df) * train_size):]
    
    return train_data, test_data
```

- **Função `split_data`**:
  - Divide o conjunto de dados em dois subconjuntos: treinamento e teste.
  - **`train_size`**: Define a proporção de dados a serem usados para treinamento (80% por padrão).
  - Retorna dois DataFrames: um para treinamento e outro para teste.

### Execução Principal

```python
if __name__ == "__main__":
    try:
        # Carregar dados limpos
        btc_data = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Arquivo de dados limpo não encontrado: {input_file}. Verifique se o script explore_clean_data.py foi executado corretamente.")
        exit(1)
    
    # Criar novos recursos
    btc_data_with_features = create_features(btc_data)
    
    # Normalizar os dados
    btc_data_normalized = normalize_data(btc_data_with_features)
    
    # Dividir os dados em conjuntos de treinamento e teste
    train_data, test_data = split_data(btc_data_normalized)
    
    # Salvar os conjuntos de dados
    train_data.to_csv(output_file_train, index=False)
    test_data.to_csv(output_file_test, index=False)
    
    print(f"Conjuntos de dados de treino e teste salvos com sucesso em {output_file_train} e {output_file_test}!")
```

- **Carregamento dos Dados Limpos**:
  - Carrega os dados limpos do arquivo CSV. Se o arquivo não for encontrado, imprime uma mensagem de erro e encerra o script.

- **Criação de Novos Recursos**:
  - Utiliza a função `create_features` para criar novas features a partir dos dados originais.

- **Normalização dos Dados**:
  - Normaliza os dados usando a função `normalize_data`.

- **Divisão dos Dados**:
  - Divide o conjunto de dados normalizados em subconjuntos de treinamento e teste usando a função `split_data`.

- **Salvamento dos Dados**:
  - Salva os conjuntos de dados de treino e teste em arquivos CSV separados.

### Resumo

Este script executa várias etapas de preparação dos dados para um modelo de aprendizado de máquina:

1. **Criação de Novos Recursos**: Gera novas features úteis para o modelo, como médias móveis, RSI e momentum.
2. **Normalização dos Dados**: Escala os dados para um intervalo entre 0 e 1, o que melhora a performance dos algoritmos.
3. **Divisão dos Dados**: Separa os dados em conjuntos de treinamento e teste para validar o modelo.
4. **Salvamento dos Dados Preparados**: Exporta os dados preparados para arquivos CSV, prontos para serem utilizados no treinamento de modelos.

A modularidade do código permite fácil reutilização e adaptação para diferentes conjuntos de dados ou ajustes nos parâmetros.