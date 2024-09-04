import pytest
from dash import Dash
from app.app import app

def test_dash_app():
    # Testa se o aplicativo Dash é criado e possui o layout correto
    assert isinstance(app, Dash)
    assert app.layout is not None
