import dash
from dash import dcc, html
import plotly.express as px
import plotly.graph_objects as go
from flask import Flask
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
from apscheduler.schedulers.background import BackgroundScheduler
import pytz
import backtrader as bt
import os

# Inicializar o servidor Flask
server = Flask(__name__)

# Variável global para armazenar os dados mais recentes
latest_data = pd.DataFrame()

# Função para converter o horário de UTC para o horário de Brasília
def convert_to_brasilia_time(utc_time):
    brasilia_tz = pytz.timezone('America/Sao_Paulo')
    return utc_time.astimezone(brasilia_tz)

# Função para coletar dados em tempo real da API do Yahoo Finance
def fetch_real_time_data():
    global latest_data
    symbol = os.getenv('SYMBOL', 'BTC-USD')
    data = yf.download(tickers=symbol, period='1d', interval='1m')
    data.reset_index(inplace=True)
    
    # Converte a coluna 'Datetime' para o horário de Brasília
    data['Datetime'] = pd.to_datetime(data['Datetime'])
    data['Datetime'] = data['Datetime'].apply(convert_to_brasilia_time)

    latest_data = data
    print("Dados atualizados:", latest_data.tail())

# Agendar a coleta de dados a cada 1 minuto
scheduler = BackgroundScheduler()
scheduler.add_job(fetch_real_time_data, 'interval', minutes=1)
scheduler.start()

# Carregar dados iniciais
fetch_real_time_data()

# Integrar Dash com Flask
app = dash.Dash(__name__, server=server, url_base_pathname='/dashboard/')

# Definição da estratégia de Backtrader
class SmaCross(bt.Strategy):
    params = (('sma1_period', 10), ('sma2_period', 30),)

    def __init__(self):
        self.sma1 = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.sma1_period)
        self.sma2 = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.sma2_period)
        self.crossover = bt.indicators.CrossOver(self.sma1, self.sma2)  # Cruzamento de SMA
        self.stop_loss = 0.03  # Stop Loss de 3%
        self.take_profit = 0.05  # Take Profit de 5%

    def next(self):
        # Se ainda não estamos comprados e houve cruzamento para cima, compramos
        if not self.position and self.crossover > 0:
            self.size = self.broker.get_cash() // self.data.close[0]  # Tamanho da posição baseado no saldo disponível
            self.buy(size=self.size)
            print(f"Compra executada em: {self.data.datetime.date(0)}, Preço: {self.data.close[0]}")

        # Se estamos comprados e houve cruzamento para baixo, vendemos
        elif self.position:
            if self.crossover < 0:  # Condição de venda com cruzamento para baixo
                self.sell(size=self.size)
                print(f"Venda executada em: {self.data.datetime.date(0)}, Preço: {self.data.close[0]}")

            # Check para Stop-Loss ou Take-Profit
            elif self.position.size > 0:
                if self.data.close[0] <= self.position.price * (1 - self.stop_loss):
                    self.sell(size=self.size)
                    print(f"Venda por Stop-Loss em: {self.data.datetime.date(0)}, Preço: {self.data.close[0]}")
                elif self.data.close[0] >= self.position.price * (1 + self.take_profit):
                    self.sell(size=self.size)
                    print(f"Venda por Take-Profit em: {self.data.datetime.date(0)}, Preço: {self.data.close[0]}")

# Função para realizar o backtest com saldo simulado usando Backtrader
def run_backtest():
    global latest_data
    # Verificar se há dados suficientes para o backtest
    if latest_data.empty or len(latest_data) < 30:
        print("Dados insuficientes para o backtest.")
        return None, 0

    # Converta a coluna de data diretamente em um índice
    latest_data['Date'] = pd.to_datetime(latest_data['Datetime'])
    latest_data.set_index('Date', inplace=True)
    
    # Prepare dados para o Backtrader
    df_bt = bt.feeds.PandasData(dataname=latest_data)

    # Inicialize o Cerebro de Backtrader
    cerebro = bt.Cerebro()
    cerebro.addstrategy(SmaCross)
    cerebro.adddata(df_bt)

    # Defina saldo inicial e comissão
    cerebro.broker.setcash(10000.0)
    cerebro.broker.setcommission(commission=0.001)  # Ajuste de comissão

    # Execute o Backtest
    initial_cash = cerebro.broker.getvalue()
    cerebro.run()
    final_cash = cerebro.broker.getvalue()

    # Calcular lucro
    profit = final_cash - initial_cash
    print(f"Saldo inicial: {initial_cash}")
    print(f"Saldo final: {final_cash}")
    print(f"Lucro: {profit}")

    # Extrair resultados para plotagem
    df_result = pd.DataFrame({'Date': latest_data.index, 'Close': latest_data['Close'], 'Cash': cerebro.broker.getvalue()})
    backtest_fig = px.line(df_result, x='Date', y='Cash', title='Simulação de Backtest com Saldo Simulado')

    return backtest_fig, profit

# Função para atualizar os gráficos com os dados mais recentes
def update_graphs():
    global latest_data
    latest_data['Date'] = pd.to_datetime(latest_data['Datetime'])

    # Cálculo de SMA e RSI
    latest_data['SMA_10'] = latest_data['Close'].rolling(window=10).mean()
    latest_data['SMA_30'] = latest_data['Close'].rolling(window=30).mean()

    delta = latest_data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    latest_data['RSI'] = 100 - (100 / (1 + rs))

    # Correção do Gráfico de Correlação
    correlation_features = latest_data[['Close', 'SMA_10', 'SMA_30', 'RSI']].dropna()
    corr = correlation_features.corr()
    heatmap_fig = px.imshow(corr, title='Mapa de Correlação entre Variáveis', labels={'color': 'Correlação'})

    # Gráficos existentes
    line_fig = px.line(latest_data, x='Datetime', y='Close', title='Evolução do Preço do Bitcoin', labels={'Datetime': 'Data', 'Close': 'Preço de Fechamento'})

    candlestick_fig = go.Figure(data=[go.Candlestick(x=latest_data['Datetime'],
                                                     open=latest_data['Open'],
                                                     high=latest_data['High'],
                                                     low=latest_data['Low'],
                                                     close=latest_data['Close'],
                                                     increasing_line_color='green', decreasing_line_color='red')])
    candlestick_fig.update_layout(title='Gráfico de Candlestick do Bitcoin', xaxis_title='Data', yaxis_title='Preço')

    ma_fig = px.line(latest_data, x='Datetime', y=['Close', 'SMA_10', 'SMA_30'], title='Preço de Fechamento e Médias Móveis', labels={'value': 'Preço', 'Datetime': 'Data'})
    ma_fig.update_traces(marker=dict(size=2))

    volume_fig = px.bar(latest_data, x='Datetime', y='Volume', title='Volume de Negociação do Bitcoin', labels={'Datetime': 'Data', 'Volume': 'Volume de Negociação'})

    rsi_fig = px.line(latest_data, x='Datetime', y='RSI', title='Índice de Força Relativa (RSI)', labels={'Datetime': 'Data', 'RSI': 'RSI'})
    rsi_fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Sobrecomprado", annotation_position="top left")
    rsi_fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Sobrevendido", annotation_position="bottom right")

    latest_data['EMA_12'] = latest_data['Close'].ewm(span=12, adjust=False).mean()
    latest_data['EMA_26'] = latest_data['Close'].ewm(span=26, adjust=False).mean()
    latest_data['MACD'] = latest_data['EMA_12'] - latest_data['EMA_26']
    latest_data['Signal Line'] = latest_data['MACD'].ewm(span=9, adjust=False).mean()
    macd_fig = px.line(latest_data, x='Datetime', y=['MACD', 'Signal Line'], title='MACD e Linha de Sinal', labels={'value': 'Valor', 'Datetime': 'Data'})

    latest_data['Upper Band'] = latest_data['SMA_10'] + (2 * latest_data['Close'].rolling(window=10).std())
    latest_data['Lower Band'] = latest_data['SMA_10'] - (2 * latest_data['Close'].rolling(window=10).std())
    bollinger_fig = px.line(latest_data, x='Datetime', y=['Close', 'Upper Band', 'Lower Band'], title='Bollinger Bands', labels={'value': 'Preço', 'Datetime': 'Data'})

    latest_data['Momentum'] = latest_data['Close'].diff(4)
    momentum_fig = px.line(latest_data, x='Datetime', y='Momentum', title='Momentum do Preço do Bitcoin', labels={'Datetime': 'Data', 'Momentum': 'Momentum'})

    latest_data['CumVolume'] = latest_data['Volume'].cumsum()
    latest_data['CumCloseVol'] = (latest_data['Close'] * latest_data['Volume']).cumsum()
    latest_data['VWAP'] = latest_data['CumCloseVol'] / latest_data['CumVolume']
    vwap_fig = px.line(latest_data, x='Datetime', y='VWAP', title='VWAP - Volume Weighted Average Price', labels={'Datetime': 'Data', 'VWAP': 'VWAP'})

    latest_data['Date_ordinal'] = pd.to_datetime(latest_data['Datetime']).map(pd.Timestamp.toordinal)
    X = latest_data[['Date_ordinal']]
    y = latest_data['Close']
    regression_model = LinearRegression().fit(X, y)
    latest_data['Predicted_Close'] = regression_model.predict(X)
    regression_fig = px.scatter(latest_data, x='Datetime', y='Close', title='Previsão do Preço Usando Regressão Linear')
    regression_fig.add_trace(go.Scatter(x=latest_data['Datetime'], y=latest_data['Predicted_Close'], mode='lines', name='Linha de Regressão'))

    # Adicionar resultado do backtest
    backtest_fig, report = run_backtest()

    return line_fig, candlestick_fig, heatmap_fig, ma_fig, volume_fig, rsi_fig, macd_fig, bollinger_fig, momentum_fig, vwap_fig, regression_fig, backtest_fig

# Configurar o layout do Dash
app.layout = html.Div([
    html.H1("Dashboard de Análise Preditiva do Bitcoin"),
    dcc.Graph(id='line-fig'),
    dcc.Graph(id='candlestick-fig'),
    dcc.Graph(id='heatmap-fig'),
    dcc.Graph(id='ma-fig'),
    dcc.Graph(id='volume-fig'),
    dcc.Graph(id='rsi-fig'),
    dcc.Graph(id='macd-fig'),
    dcc.Graph(id='bollinger-fig'),
    dcc.Graph(id='momentum-fig'),
    dcc.Graph(id='vwap-fig'),
    dcc.Graph(id='regression-fig'),
    dcc.Graph(id='backtest-fig'),  # Novo gráfico para o backtest
    dcc.Interval(id='interval-component', interval=10*1000, n_intervals=0)  # Atualiza a cada 10 segundos
])

# Callback para atualizar os gráficos
@app.callback(
    [dash.dependencies.Output('line-fig', 'figure'),
     dash.dependencies.Output('candlestick-fig', 'figure'),
     dash.dependencies.Output('heatmap-fig', 'figure'),
     dash.dependencies.Output('ma-fig', 'figure'),
     dash.dependencies.Output('volume-fig', 'figure'),
     dash.dependencies.Output('rsi-fig', 'figure'),
     dash.dependencies.Output('macd-fig', 'figure'),
     dash.dependencies.Output('bollinger-fig', 'figure'),
     dash.dependencies.Output('momentum-fig', 'figure'),
     dash.dependencies.Output('vwap-fig', 'figure'),
     dash.dependencies.Output('regression-fig', 'figure'),
     dash.dependencies.Output('backtest-fig', 'figure')],
    [dash.dependencies.Input('interval-component', 'n_intervals')]
)
def update_figures(n_intervals):
    return update_graphs()

if __name__ == '__main__':
    server.run(debug=True, host='0.0.0.0', port=5000)
