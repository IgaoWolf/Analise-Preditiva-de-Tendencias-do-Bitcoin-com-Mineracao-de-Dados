import pytest
from scripts.modeling.train_models import train_model

def test_train_model():
    # Verifica se o modelo é treinado corretamente e retorna um objeto de modelo
    model, metrics = train_model()
    assert model is not None
    assert 'accuracy' in metrics
    assert metrics['accuracy'] > 0.6  # Exemplo: uma acurácia mínima esperada
