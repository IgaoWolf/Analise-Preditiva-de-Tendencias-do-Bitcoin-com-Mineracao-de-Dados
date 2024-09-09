### Importações

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
```

- **`pandas`** (`pd`): Biblioteca para manipulação e análise de dados estruturados.
- **`matplotlib.pyplot`** (`plt`): Biblioteca para criação de gráficos e visualizações.
- **`seaborn`** (`sns`): Biblioteca baseada no `matplotlib` que oferece visualizações estatísticas mais sofisticadas.
- **`os`**: Biblioteca padrão do Python para interagir com o sistema operacional, utilizada aqui para manipular diretórios e arquivos.

### Definição de Diretórios

```python
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
output_dir = os.path.join(base_dir, 'data/processed/')
```

- **`base_dir`**: Define o diretório base do projeto, que é obtido usando o caminho do arquivo atual (`__file__`) e subindo dois níveis na hierarquia de diretórios (`../../`).
- **`output_dir`**: Define o caminho para o diretório onde os dados processados serão salvos. Ele está configurado para o subdiretório `processed` dentro da pasta `data`.

### Verificação e Criação do Diretório de Saída

```python
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
```

- Esta parte do código verifica se o diretório de saída (`output_dir`) já existe. Se não existir, ele é criado usando `os.makedirs`.

### Caminho para o Arquivo CSV de Saída

```python
output_file = os.path.join(output_dir, 'clean_btc_data.csv')
```

- **`output_file`**: Define o caminho completo para o arquivo CSV que conterá os dados limpos do Bitcoin. Este arquivo será salvo no diretório `processed`.

### Função para Carregar Dados

```python
def load_data(file_path):
    """
    Carrega o conjunto de dados de um arquivo CSV.

    Parâmetros:
    - file_path (str): Caminho para o arquivo CSV.

    Retorno:
    - DataFrame do pandas com os dados carregados.
    """
    return pd.read_csv(file_path)
```

- **Função `load_data`**:
  - Carrega um arquivo CSV especificado por `file_path` e retorna um DataFrame do `pandas` contendo os dados.

### Função para Exploração de Dados

```python
def explore_data(df):
    """
    Realiza uma análise exploratória inicial dos dados.

    Parâmetros:
    - df (DataFrame): Conjunto de dados.

    Retorno:
    - Nenhum. Apenas exibe informações e gráficos.
    """
    print("Primeiras 5 linhas dos dados:")
    print(df.head())

    print("\nInformações gerais dos dados:")
    print(df.info())

    print("\nDescrição estatística dos dados:")
    print(df.describe())

    # Gráfico de linhas para preço de fechamento
    plt.figure(figsize=(10, 6))
    plt.plot(df['Date'], df['Close'], label='Preço de Fechamento')
    plt.xlabel('Data')
    plt.ylabel('Preço de Fechamento (USD)')
    plt.title('Histórico do Preço de Fechamento do BTC')
    plt.legend()
    plt.show()
```

- **Função `explore_data`**:
  - Recebe um DataFrame (`df`) e realiza uma análise exploratória inicial:
    - Imprime as primeiras 5 linhas do DataFrame (`df.head()`).
    - Exibe informações gerais sobre os dados, como tipos de dados e valores ausentes (`df.info()`).
    - Fornece uma descrição estatística dos dados numéricos, como média, desvio padrão, valores mínimos e máximos (`df.describe()`).
    - Cria um gráfico de linhas para visualizar o histórico do preço de fechamento do Bitcoin.

### Função para Limpeza de Dados

```python
def clean_data(df):
    """
    Limpa o conjunto de dados removendo valores ausentes e anômalos.

    Parâmetros:
    - df (DataFrame): Conjunto de dados original.

    Retorno:
    - DataFrame limpo.
    """
    # Converter a coluna 'Date' para datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Remover linhas com valores ausentes
    df = df.dropna()

    # Remover valores anômalos (opcional: remover outliers)
    # Por exemplo, remover preços negativos
    df = df[df['Close'] > 0]

    return df
```

- **Função `clean_data`**:
  - Recebe um DataFrame (`df`) e realiza as seguintes etapas de limpeza:
    - Converte a coluna 'Date' para o tipo `datetime` para facilitar a manipulação de datas.
    - Remove quaisquer linhas que contenham valores ausentes (`df.dropna()`).
    - Remove dados anômalos, como preços de fechamento negativos.

### Execução Principal

```python
if __name__ == "__main__":
    # Caminho para o arquivo CSV dos dados brutos do BTC
    input_file = os.path.join(base_dir, 'data/raw/btc_data.csv')
    
    try:
        # Carrega os dados
        btc_data = load_data(input_file)
    except FileNotFoundError:
        print(f"Arquivo de dados bruto não encontrado: {input_file}. Verifique se o script de coleta foi executado corretamente.")
        exit(1)
    
    # Explora os dados
    explore_data(btc_data)
    
    # Limpa os dados
    clean_btc_data = clean_data(btc_data)
    
    # Salva o conjunto de dados limpo
    clean_btc_data.to_csv(output_file, index=False)
    print(f"Dados limpos salvos com sucesso em {output_file}!")
```

- **`if __name__ == "__main__"`**:
  - Verifica se o script está sendo executado diretamente.
  - Define o caminho para o arquivo de dados brutos (`input_file`).
  - Tenta carregar os dados usando a função `load_data()`. Se o arquivo não for encontrado, imprime uma mensagem de erro e encerra o script.
  - Realiza a exploração inicial dos dados com `explore_data()`.
  - Limpa os dados utilizando `clean_data()`.
  - Salva o DataFrame limpo em um arquivo CSV no caminho definido por `output_file`.

### Resumo

Este script realiza várias etapas importantes para manipulação de dados históricos de preços do Bitcoin:
1. **Carregamento dos Dados**: Importa dados brutos de um arquivo CSV.
2. **Exploração dos Dados**: Gera estatísticas descritivas e gráficos para entender melhor o conjunto de dados.
3. **Limpeza dos Dados**: Remove valores ausentes e anômalos para garantir a qualidade dos dados.
4. **Salvamento dos Dados Limpos**: Salva os dados processados em um novo arquivo CSV para uso posterior.

A modularidade do código permite fácil reutilização e adaptação para outros conjuntos de dados financeiros ou períodos de análise.