import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Diretório base do projeto
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

# Caminho para o diretório de saída
output_dir = os.path.join(base_dir, 'data/processed/')

# Verifica se o diretório de saída existe; se não, cria o diretório
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Caminho para o arquivo CSV dos dados limpos
output_file = os.path.join(output_dir, 'clean_btc_data.csv')

def load_data(file_path):
    """
    Carrega o conjunto de dados de um arquivo CSV.

    Parâmetros:
    - file_path (str): Caminho para o arquivo CSV.

    Retorno:
    - DataFrame do pandas com os dados carregados.
    """
    return pd.read_csv(file_path)

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
