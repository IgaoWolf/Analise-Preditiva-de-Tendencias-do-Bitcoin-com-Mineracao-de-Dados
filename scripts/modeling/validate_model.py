import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os
import joblib  # Importa joblib para salvar o modelo

# Diretório base do projeto
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

# Caminho para os arquivos de treino e teste
train_file = os.path.join(base_dir, 'data/processed/train_data.csv')
test_file = os.path.join(base_dir, 'data/processed/test_data.csv')

# Caminho para salvar o modelo otimizado
model_dir = os.path.join(base_dir, 'models')
model_save_path = os.path.join(model_dir, 'best_model.pkl')

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

    # Melhor modelo encontrado anteriormente
    best_model = MLPClassifier(activation='tanh', alpha=0.0001, hidden_layer_sizes=(50, 50), 
                               learning_rate='constant', max_iter=2000, solver='adam')

    # Validando o modelo com K-Fold Cross-Validation
    print("Iniciando a Validação Cruzada K-Fold...")
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_val_score(best_model, X_train, y_train, cv=kfold, scoring='accuracy')
    print(f"Acurácia média da Validação Cruzada: {cv_results.mean():.4f}")
    print(f"Desvio padrão da Acurácia: {cv_results.std():.4f}")

    # Treinando e avaliando o modelo no conjunto de teste
    train_and_evaluate_model(best_model, X_train, y_train, X_test, y_test)

    # Verificar se o diretório para salvar o modelo existe, e criar se não existir
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"Diretório criado: {model_dir}")

    # Salvar o modelo otimizado para uso em produção
    joblib.dump(best_model, model_save_path)
    print(f"Modelo otimizado salvo em: {model_save_path}")
