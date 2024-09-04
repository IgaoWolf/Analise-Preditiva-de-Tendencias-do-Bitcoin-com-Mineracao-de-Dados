import pytest
from scripts.modeling.optimize_model import optimize_model

def test_optimize_model():
    # Testa se o modelo é otimizado corretamente
    best_model = optimize_model()
    assert best_model is not None
    assert hasattr(best_model, 'predict')  # Verifica se o modelo otimizado tem método de previsão
