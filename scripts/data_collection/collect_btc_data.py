import yfinance as yf
import os

# Defina o símbolo da criptomoeda (BTC-USD para Bitcoin)
btc_symbol = 'BTC-USD'

# Defina o período para o qual deseja coletar os dados (exemplo: 5 anos)
period = '5y'

# Caminho para salvar o arquivo CSV na pasta correta
output_dir = '../../data/raw/'
output_file = os.path.join(output_dir, 'btc_data.csv')

def collect_btc_data(symbol, period, output_path):
    """
    Coleta dados históricos de uma criptomoeda a partir da API do Yahoo Finance e salva em um arquivo CSV.

    Parâmetros:
    - symbol (str): Símbolo da criptomoeda (ex.: 'BTC-USD').
    - period (str): Período de dados a serem coletados (ex.: '5y' para 5 anos).
    - output_path (str): Caminho para salvar o arquivo CSV.
    """
    # Baixa os dados do Yahoo Finance
    btc_data = yf.download(tickers=symbol, period=period, interval='1d')

    # Salva os dados em um arquivo CSV
    btc_data.to_csv(output_path)
    print(f"Dados do {symbol} coletados e salvos com sucesso em {output_path}!")

if __name__ == "__main__":
    # Coleta os dados de BTC e salva no arquivo CSV
    collect_btc_data(btc_symbol, period, output_file)
