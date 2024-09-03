import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

# Diretório base do projeto
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

# Caminho para o arquivo CSV dos dados limpos do BTC
input_file = os.path.join(base_dir, 'data/processed/clean_btc_data.csv')

# Caminhos para os arquivos de saída de treino e teste
output_file_train = os.path.join(base_dir, 'data/processed/train_data.csv')
output_file_test = os.path.join(base_dir, 'data/processed/test_data.csv')

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
