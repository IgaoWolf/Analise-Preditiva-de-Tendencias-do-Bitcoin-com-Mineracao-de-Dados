import dash
from dash import dcc, html
import plotly.express as px
import plotly.graph_objects as go
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import yfinance as yf
from sklearn.linear_model import LinearRegression
from apscheduler.schedulers.background import BackgroundScheduler
import pytz  # Importar pytz para conversão de fuso horário

# Inicializar o servidor Flask
server = Flask(__name__)

# Carregar o modelo treinado
model_path = 'models/best_model.pkl'
model = joblib.load(model_path)

# Variável global para armazenar os dados mais recentes
latest_data = pd.DataFrame()

# Função para converter o horário de UTC para o horário de Brasília
def convert_to_brasilia_time(utc_time):
    brasilia_tz = pytz.timezone('America/Sao_Paulo')
    return utc_time.astimezone(brasilia_tz)

# Função para coletar dados em tempo real da API do Yahoo Finance
def fetch_real_time_data():
    global latest_data
    symbol = "BTC-USD"  # Exemplo: símbolo do Bitcoin
    data = yf.download(tickers=symbol, period='1d', interval='1m')  # Coletar dados de 1 dia em intervalos de 1 minuto
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

# Função de predição em tempo real
@server.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({'status': 'failure', 'error': 'No data provided'}), 400

    try:
        input_data = pd.DataFrame([data])
        expected_features = ['Adj Close', 'High', 'Low', 'Volume', 'EMA_5', 'EMA_10']
        missing_features = [feature for feature in expected_features if feature not in input_data.columns]
        if missing_features:
            return jsonify({'status': 'failure', 'error': f'Missing features: {missing_features}'}), 400

        prediction = model.predict(input_data)
        return jsonify({'status': 'success', 'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'status': 'failure', 'error': str(e)}), 500

# Integrar Dash com Flask
app = dash.Dash(__name__, server=server, url_base_pathname='/dashboard/')

# Função para atualizar os gráficos com os dados mais recentes
def update_graphs():
    global latest_data
    df = latest_data.copy()
    df['Date'] = pd.to_datetime(df['Datetime'])

    # Gráfico de linha para a evolução do preço do Bitcoin
    line_fig = px.line(df, x='Date', y='Close', title='Evolução do Preço do Bitcoin', labels={'Date': 'Data', 'Close': 'Preço de Fechamento'})

    # Gráfico de candlestick para visualizar os movimentos de mercado
    candlestick_fig = go.Figure(data=[go.Candlestick(x=df['Date'],
                                                     open=df['Open'],
                                                     high=df['High'],
                                                     low=df['Low'],
                                                     close=df['Close'],
                                                     increasing_line_color='green', decreasing_line_color='red')])
    candlestick_fig.update_layout(title='Gráfico de Candlestick do Bitcoin', xaxis_title='Data', yaxis_title='Preço')

    # Heatmap de correlação entre variáveis numéricas
    numerical_df = df.select_dtypes(include=['float64', 'int64'])
    corr = numerical_df.corr()
    heatmap_fig = px.imshow(corr, title='Mapa de Correlação entre Variáveis', labels={'color': 'Correlação'})

    # Gráfico de Média Móvel
    df['SMA_10'] = df['Close'].rolling(window=10).mean()  # Média móvel de 10 dias
    df['SMA_30'] = df['Close'].rolling(window=30).mean()  # Média móvel de 30 dias
    ma_fig = px.line(df, x='Date', y=['Close', 'SMA_10', 'SMA_30'], title='Preço de Fechamento e Médias Móveis', labels={'value': 'Preço', 'Date': 'Data'})
    ma_fig.update_traces(marker=dict(size=2))

    # Gráfico de Volume de Negociação
    volume_fig = px.bar(df, x='Date', y='Volume', title='Volume de Negociação do Bitcoin', labels={'Date': 'Data', 'Volume': 'Volume de Negociação'})

    # Gráfico de RSI (Índice de Força Relativa)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    rsi_fig = px.line(df, x='Date', y='RSI', title='Índice de Força Relativa (RSI)', labels={'Date': 'Data', 'RSI': 'RSI'})
    rsi_fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Sobrecomprado", annotation_position="top left")
    rsi_fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Sobrevendido", annotation_position="bottom right")

    # Gráfico de Comparação de Preços de Abertura e Fechamento
    price_comparison_fig = px.bar(df, x='Date', y=[df['Close'] - df['Open']], title='Diferença de Preços de Abertura e Fechamento', labels={'value': 'Diferença de Preço', 'Date': 'Data'})

    # Gráfico de Regressão Linear
    df['Date_ordinal'] = pd.to_datetime(df['Date']).map(pd.Timestamp.toordinal)
    X = df[['Date_ordinal']]
    y = df['Close']
    regression_model = LinearRegression().fit(X, y)
    df['Predicted_Close'] = regression_model.predict(X)
    regression_fig = px.scatter(df, x='Date', y='Close', title='Previsão do Preço Usando Regressão Linear')
    regression_fig.add_trace(go.Scatter(x=df['Date'], y=df['Predicted_Close'], mode='lines', name='Linha de Regressão'))

    return line_fig, candlestick_fig, heatmap_fig, ma_fig, volume_fig, rsi_fig, price_comparison_fig, regression_fig

# Configurar o layout do Dash
app.layout = html.Div([
    html.H1("Dashboard de Análise Preditiva do Bitcoin"),
    dcc.Graph(id='line-fig'),
    dcc.Graph(id='candlestick-fig'),
    dcc.Graph(id='heatmap-fig'),
    dcc.Graph(id='ma-fig'),
    dcc.Graph(id='volume-fig'),
    dcc.Graph(id='rsi-fig'),
    dcc.Graph(id='price-comparison-fig'),
    dcc.Graph(id='regression-fig'),
    dcc.Interval(id='interval-component', interval=30*1000, n_intervals=0)  # Atualiza a cada 30 segundos
])

# Callback para atualizar os gráficos
@app.callback(
    [dash.dependencies.Output('line-fig', 'figure'),
     dash.dependencies.Output('candlestick-fig', 'figure'),
     dash.dependencies.Output('heatmap-fig', 'figure'),
     dash.dependencies.Output('ma-fig', 'figure'),
     dash.dependencies.Output('volume-fig', 'figure'),
     dash.dependencies.Output('rsi-fig', 'figure'),
     dash.dependencies.Output('price-comparison-fig', 'figure'),
     dash.dependencies.Output('regression-fig', 'figure')],
    [dash.dependencies.Input('interval-component', 'n_intervals')]
)
def update_figures(n_intervals):
    return update_graphs()

if __name__ == '__main__':
    server.run(debug=True, host='0.0.0.0', port=5000)
