### Importações
```python
import yfinance as yf
import os
```

- `yfinance` é uma biblioteca Python que permite acessar dados financeiros do Yahoo Finance. Aqui, é usada para baixar dados históricos de uma criptomoeda.
- `os` é uma biblioteca padrão do Python que fornece funções para interagir com o sistema operacional. Neste caso, é usada para manipular caminhos de arquivos.

### Definição de Variáveis

```python
btc_symbol = 'BTC-USD'
```

- `btc_symbol`: Uma string que representa o símbolo da criptomoeda Bitcoin (BTC) em relação ao Dólar Americano (USD) no Yahoo Finance. O símbolo `'BTC-USD'` é utilizado para especificar a criptomoeda que será coletada.

```python
period = '5y'
```

- `period`: Define o período de tempo para o qual os dados históricos serão coletados. Neste caso, `'5y'` significa que os dados dos últimos 5 anos serão baixados.

```python
output_dir = '../../data/raw/'
output_file = os.path.join(output_dir, 'btc_data.csv')
```

- `output_dir`: O caminho relativo da pasta onde o arquivo CSV será salvo. Neste exemplo, está configurado para um diretório chamado `raw` dentro de `data`.
- `output_file`: Usa a função `os.path.join` para combinar `output_dir` e `'btc_data.csv'` em um caminho completo para o arquivo de saída onde os dados serão salvos.

### Função para Coletar Dados

```python
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
```

- **Função `collect_btc_data`**:
  - Esta função recebe três parâmetros:
    - `symbol`: O símbolo da criptomoeda que queremos coletar.
    - `period`: O período de tempo para o qual queremos coletar os dados.
    - `output_path`: O caminho onde o arquivo CSV resultante será salvo.
  - **Uso do `yf.download`**:
    - Esta linha baixa os dados históricos da criptomoeda especificada. 
    - `tickers=symbol`: Especifica o símbolo da criptomoeda.
    - `period=period`: Define o período para o qual queremos coletar os dados.
    - `interval='1d'`: Define que os dados coletados serão em intervalos diários.
  - **Salvando Dados em CSV**:
    - `btc_data.to_csv(output_path)`: Salva o DataFrame contendo os dados coletados em um arquivo CSV no caminho especificado por `output_path`.
  - **Mensagem de Sucesso**:
    - `print(...)`: Exibe uma mensagem informando que os dados foram coletados e salvos com sucesso.

### Execução Principal

```python
if __name__ == "__main__":
    # Coleta os dados de BTC e salva no arquivo CSV
    collect_btc_data(btc_symbol, period, output_file)
```

- **`if __name__ == "__main__"`**:
  - Esta linha verifica se o script está sendo executado diretamente (e não importado como um módulo em outro script).
  - Se for o caso, ele chama a função `collect_btc_data` com os parâmetros `btc_symbol`, `period` e `output_file` previamente definidos, coletando os dados do Bitcoin e salvando-os no arquivo CSV.

