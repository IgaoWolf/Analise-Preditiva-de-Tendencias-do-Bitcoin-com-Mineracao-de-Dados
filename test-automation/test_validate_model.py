import pytest
from scripts.modeling.validate_model import validate_model

def test_validate_model():
    # Testa se a validação do modelo retorna métricas corretas
    validation_results = validate_model()
    assert validation_results is not None
    assert 'cross_val_score' in validation_results
