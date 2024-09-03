import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os

# Diretório base do projeto
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

# Caminho para os arquivos de treino e teste
train_file = os.path.join(base_dir, 'data/processed/train_data.csv')
test_file = os.path.join(base_dir, 'data/processed/test_data.csv')

def load_data(train_path, test_path):
    """
    Carrega os conjuntos de dados de treino e teste.

    Parâmetros:
    - train_path (str): Caminho para o arquivo de treino.
    - test_path (str): Caminho para o arquivo de teste.

    Retorno:
    - Dois DataFrames do pandas: dados de treino e dados de teste.
    """
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    return train_data, test_data

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Treina e avalia um modelo de aprendizado de máquina.

    Parâmetros:
    - model: Algoritmo de aprendizado de máquina a ser treinado.
    - X_train: Dados de entrada de treino.
    - y_train: Rótulos de treino.
    - X_test: Dados de entrada de teste.
    - y_test: Rótulos de teste.

    Retorno:
    - Exibe a acurácia, a matriz de confusão e o relatório de classificação do modelo.
    """
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    print(f"\nModelo: {model.__class__.__name__} (Otimizado)")
    print("Acurácia:", accuracy_score(y_test, predictions))
    print("Matriz de Confusão:\n", confusion_matrix(y_test, predictions))
    print("Relatório de Classificação:\n", classification_report(y_test, predictions))

if __name__ == "__main__":
    # Carrega os dados de treino e teste
    train_data, test_data = load_data(train_file, test_file)

    # Separar as features (X) e o target (y)
    X_train = train_data.drop(['Date', 'Close'], axis=1)  # Remove as colunas 'Date' e 'Close'
    y_train = train_data['Close'] > train_data['Close'].shift(1)  # Define o target como subida ou descida
    X_test = test_data.drop(['Date', 'Close'], axis=1)
    y_test = test_data['Close'] > test_data['Close'].shift(1)

    # Ajuste dos parâmetros para o Grid Search com mais iterações
    param_grid = {
        'hidden_layer_sizes': [(100,), (50, 50), (100, 50)],  # Testa diferentes tamanhos de camadas ocultas
        'activation': ['tanh', 'relu'],  # Testa diferentes funções de ativação
        'solver': ['adam'],  # Usa o otimizador 'adam'
        'alpha': [0.0001, 0.001],  # Regularização
        'learning_rate': ['constant', 'adaptive'],  # Diferentes métodos de taxa de aprendizado
        'max_iter': [2000]  # Aumenta o número de iterações para melhorar a convergência
    }

    # Configurando o Grid Search
    grid_search = GridSearchCV(MLPClassifier(), param_grid, n_jobs=-1, cv=3)

    # Executando o Grid Search para encontrar os melhores parâmetros
    print("Iniciando o Grid Search para otimizar o MLPClassifier com mais iterações...")
    grid_search.fit(X_train, y_train)

    # Exibindo os melhores parâmetros encontrados
    print("Melhores parâmetros encontrados:", grid_search.best_params_)

    # Avaliando o melhor modelo encontrado no Grid Search
    best_model = grid_search.best_estimator_
    train_and_evaluate_model(best_model, X_train, y_train, X_test, y_test)
