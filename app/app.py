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
    symbol = "BTC-USD"
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
class SmaCross(bt.SignalStrategy):
    def __init__(self):
        sma1 = bt.indicators.SimpleMovingAverage(self.data.close, period=10)
        sma2 = bt.indicators.SimpleMovingAverage(self.data.close, period=30)
        self.signal_add(bt.SIGNAL_LONG, bt.ind.CrossOver(sma1, sma2))

# Função para realizar o backtest com saldo simulado usando Backtrader
def run_backtest():
    global latest_data
    df = latest_data.copy()
    df['Date'] = pd.to_datetime(df['Datetime'])
    
    # Preparar dados para o Backtrader
    df.set_index('Date', inplace=True)
    df_bt = bt.feeds.PandasData(dataname=df)

    # Inicializar o Cerebro de Backtrader
    cerebro = bt.Cerebro()
    cerebro.addstrategy(SmaCross)
    cerebro.adddata(df_bt)

    # Definir saldo inicial
    cerebro.broker.setcash(10000.0)

    # Executar o Backtest
    initial_cash = cerebro.broker.getvalue()
    cerebro.run()
    final_cash = cerebro.broker.getvalue()

    # Obter resultados
    profit = final_cash - initial_cash
    print(f"Saldo inicial: {initial_cash}")
    print(f"Saldo final: {final_cash}")
    print(f"Lucro: {profit}")

    # Extrair os resultados do backtest para plotagem com Dash
    df_result = pd.DataFrame({'Date': df.index, 'Close': df['Close'], 'Cash': cerebro.broker.getvalue()})
    backtest_fig = px.line(df_result, x='Date', y='Cash', title='Simulação de Backtest com Saldo Simulado')

    return backtest_fig, profit

# Função para atualizar os gráficos com os dados mais recentes
def update_graphs():
    global latest_data
    df = latest_data.copy()
    df['Date'] = pd.to_datetime(df['Datetime'])

    # Calcular os indicadores antes de criar o gráfico de correlação
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_30'] = df['Close'].rolling(window=30).mean()

    # Cálculo do RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Correção do Gráfico de Correlação
    correlation_features = df[['Close', 'SMA_10', 'SMA_30', 'RSI']].dropna()
    corr = correlation_features.corr()
    heatmap_fig = px.imshow(corr, title='Mapa de Correlação entre Variáveis', labels={'color': 'Correlação'})

    # Gráficos existentes
    line_fig = px.line(df, x='Date', y='Close', title='Evolução do Preço do Bitcoin', labels={'Date': 'Data', 'Close': 'Preço de Fechamento'})

    candlestick_fig = go.Figure(data=[go.Candlestick(x=df['Date'],
                                                     open=df['Open'],
                                                     high=df['High'],
                                                     low=df['Low'],
                                                     close=df['Close'],
                                                     increasing_line_color='green', decreasing_line_color='red')])
    candlestick_fig.update_layout(title='Gráfico de Candlestick do Bitcoin', xaxis_title='Data', yaxis_title='Preço')

    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_30'] = df['Close'].rolling(window=30).mean()
    ma_fig = px.line(df, x='Date', y=['Close', 'SMA_10', 'SMA_30'], title='Preço de Fechamento e Médias Móveis', labels={'value': 'Preço', 'Date': 'Data'})
    ma_fig.update_traces(marker=dict(size=2))

    volume_fig = px.bar(df, x='Date', y='Volume', title='Volume de Negociação do Bitcoin', labels={'Date': 'Data', 'Volume': 'Volume de Negociação'})

    rsi_fig = px.line(df, x='Date', y='RSI', title='Índice de Força Relativa (RSI)', labels={'Date': 'Data', 'RSI': 'RSI'})
    rsi_fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Sobrecomprado", annotation_position="top left")
    rsi_fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Sobrevendido", annotation_position="bottom right")

    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    macd_fig = px.line(df, x='Date', y=['MACD', 'Signal Line'], title='MACD e Linha de Sinal', labels={'value': 'Valor', 'Date': 'Data'})

    df['Upper Band'] = df['SMA_10'] + (2 * df['Close'].rolling(window=10).std())
    df['Lower Band'] = df['SMA_10'] - (2 * df['Close'].rolling(window=10).std())
    bollinger_fig = px.line(df, x='Date', y=['Close', 'Upper Band', 'Lower Band'], title='Bollinger Bands', labels={'value': 'Preço', 'Date': 'Data'})

    df['Momentum'] = df['Close'].diff(4)
    momentum_fig = px.line(df, x='Date', y='Momentum', title='Momentum do Preço do Bitcoin', labels={'Date': 'Data', 'Momentum': 'Momentum'})

    df['CumVolume'] = df['Volume'].cumsum()
    df['CumCloseVol'] = (df['Close'] * df['Volume']).cumsum()
    df['VWAP'] = df['CumCloseVol'] / df['CumVolume']
    vwap_fig = px.line(df, x='Date', y='VWAP', title='VWAP - Volume Weighted Average Price', labels={'Date': 'Data', 'VWAP': 'VWAP'})

    df['Date_ordinal'] = pd.to_datetime(df['Date']).map(pd.Timestamp.toordinal)
    X = df[['Date_ordinal']]
    y = df['Close']
    regression_model = LinearRegression().fit(X, y)
    df['Predicted_Close'] = regression_model.predict(X)
    regression_fig = px.scatter(df, x='Date', y='Close', title='Previsão do Preço Usando Regressão Linear')
    regression_fig.add_trace(go.Scatter(x=df['Date'], y=df['Predicted_Close'], mode='lines', name='Linha de Regressão'))

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
     dash.dependencies.Output('backtest-fig', 'figure')],  # Saída adicional para o gráfico de backtest
    [dash.dependencies.Input('interval-component', 'n_intervals')]
)
def update_figures(n_intervals):
    return update_graphs()

if __name__ == '__main__':
    server.run(debug=True, host='0.0.0.0', port=5000)
