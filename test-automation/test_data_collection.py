import pytest
import pandas as pd
from scripts.data_collection.collect_btc_data import collect_data  # Certifique-se de que o caminho está correto

def test_collect_data():
    # Testa se a função de coleta de dados retorna um DataFrame não vazio
    df = collect_data()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty

    # Testa se o DataFrame contém colunas essenciais
    expected_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    assert all(column in df.columns for column in expected_columns)
