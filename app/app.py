import dash
from dash import dcc, html
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Carregar os dados de treino
df = pd.read_csv('data/processed/train_data.csv')

# Converter a coluna 'Date' para o formato datetime
df['Date'] = pd.to_datetime(df['Date'])

# Filtrar apenas as colunas numéricas para a correlação
numerical_df = df.select_dtypes(include=['float64', 'int64'])

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
from sklearn.linear_model import LinearRegression
import numpy as np

df['Date_ordinal'] = pd.to_datetime(df['Date']).map(pd.Timestamp.toordinal)
X = df[['Date_ordinal']]
y = df['Close']
model = LinearRegression().fit(X, y)
df['Predicted_Close'] = model.predict(X)
regression_fig = px.scatter(df, x='Date', y='Close', title='Previsão do Preço Usando Regressão Linear')
regression_fig.add_trace(go.Scatter(x=df['Date'], y=df['Predicted_Close'], mode='lines', name='Linha de Regressão'))

# Configurar o layout do Dash
app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1("Dashboard de Análise Preditiva do Bitcoin"),
    dcc.Graph(figure=line_fig),  # Gráfico de linha
    dcc.Graph(figure=candlestick_fig),  # Gráfico de candlestick
    dcc.Graph(figure=heatmap_fig),  # Heatmap de correlação
    dcc.Graph(figure=ma_fig),  # Gráfico de médias móveis
    dcc.Graph(figure=volume_fig),  # Gráfico de volume de negociação
    dcc.Graph(figure=rsi_fig),  # Gráfico de RSI
    dcc.Graph(figure=price_comparison_fig),  # Gráfico de comparação de preços
    dcc.Graph(figure=regression_fig)  # Gráfico de regressão linear
])

if __name__ == '__main__':
    app.run_server(debug=True)
