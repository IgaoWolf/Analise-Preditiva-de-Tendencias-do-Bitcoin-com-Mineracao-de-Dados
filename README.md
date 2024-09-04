https://youtu.be/9vpEJ1P3vxo

```markdown
# Análise Preditiva de Tendências do Bitcoin com Mineração de Dados

Este projeto utiliza técnicas de mineração de dados e aprendizado de máquina para prever a tendência de alta ou baixa do Bitcoin (BTC) com base em dados históricos. Utilizando dados de preço, volume de negociação e indicadores técnicos, o projeto explora a eficácia de diferentes algoritmos de aprendizado de máquina para identificar padrões que possam antecipar movimentos no mercado de criptomoedas. As visualizações são criadas no Looker Studio para facilitar a análise e interpretação dos resultados.

## Estrutura do Projeto

- `data/processed/`
  - Contém os arquivos CSV com dados processados:
    - `train_data.csv`: Conjunto de dados de treinamento.
    - `test_data.csv`: Conjunto de dados de teste.
    - `clean_btc_data.csv`: Conjunto de dados pré-processados com preços, volumes e indicadores técnicos.

- `scripts/`
  - `data_processing/`
    - **`explore_clean_data.py`**: Script para explorar e limpar os dados.
    - **`prepare_data.py`**: Script para preparar os dados para a modelagem.
  - `modeling/`
    - **`train_models.py`**: Treinamento inicial de múltiplos modelos de aprendizado de máquina.
    - **`optimize_model.py`**: Otimização dos hiperparâmetros do modelo `MLPClassifier` usando Grid Search.
    - **`validate_model.py`**: Validação cruzada do modelo otimizado e salvamento do modelo final.

- `models/`
  - Contém o arquivo do modelo otimizado salvo:
    - `best_model.pkl`: Modelo `MLPClassifier` otimizado e salvo para uso em produção.

## Instalação

1. Clone o repositório:
   ```bash
   git clone https://github.com/IgaoWolf/Analise-Preditiva-de-Tendencias-do-Bitcoin-com-Mineracao-de-Dados.git
   cd Analise-Preditiva-de-Tendencias-do-Bitcoin-com-Mineracao-de-Dados
   ```

2. Instale as dependências necessárias:
   ```bash
   pip install -r requirements.txt
   ```
## Uso

### 1. Pré-processamento e Limpeza dos Dados

Execute o script `explore_clean_data.py` para explorar e limpar os dados coletados:

```bash
python scripts/data_processing/explore_clean_data.py
```

### 2. Preparação dos Dados para Modelagem

Prepare os dados para modelagem executando o script `prepare_data.py`:

```bash
python scripts/data_processing/prepare_data.py
```

### 3. Treinamento e Otimização do Modelo

Treine o modelo usando o script `train_models.py`:

```bash
python scripts/modeling/train_models.py
```

Otimize o modelo utilizando o script `optimize_model.py`:

```bash
python scripts/modeling/optimize_model.py
```

### 4. Validação e Salvamento do Modelo

Valide o modelo e salve-o para uso em produção:

```bash
python scripts/modeling/validate_model.py
```
### 5. Visualização no Looker Studio

- Conecte os arquivos CSV (`train_data.csv` e `test_data.csv`) ao Google Sheets.
- Conecte o Google Sheets ao Looker Studio para criar visualizações de tendências e análises de mercado.

## Contribuição

Sinta-se à vontade para enviar pull requests e abrir issues para sugerir melhorias ou relatar problemas.

## Contato

Para dúvidas ou sugestões, entre em contato pelo e-mail: igaowolf@gmail.com.
