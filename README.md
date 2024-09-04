# Previsão de Tendências do Bitcoin Usando Mineração de Dados e Aprendizado de Máquina

### Lembrando que o horário está como UTC

## Resumo Executivo

Este projeto tem como objetivo prever a tendência de alta ou baixa do Bitcoin com base em dados históricos. Utilizamos técnicas de mineração de dados e aprendizado de máquina para construir um modelo preditivo, avaliar seu desempenho e implementá-lo em um ambiente onde previsões podem ser feitas com base em dados atualizados. Os métodos incluem coleta de dados da API do Yahoo Finance, exploração e limpeza dos dados, criação de features, otimização de hiperparâmetros e validação cruzada. Os resultados indicam que o modelo escolhido oferece boa precisão na previsão de tendências, mas há espaço para melhorias futuras.

## Introdução

### Objetivo do Projeto

O objetivo deste projeto é prever a tendência de alta ou baixa do Bitcoin com base em dados históricos, utilizando técnicas de mineração de dados e aprendizado de máquina.

### Contexto

A previsão de preços de criptomoedas é uma tarefa desafiadora devido à alta volatilidade do mercado. Neste contexto, o uso de técnicas de mineração de dados permite analisar grandes volumes de dados e identificar padrões que podem ser usados para prever movimentos futuros de preços.

## Metodologia

### Coleta de Dados

Os dados foram coletados utilizando a API do Yahoo Finance, que fornece informações detalhadas sobre o preço de fechamento, volume de negociação, e outras variáveis importantes para o período analisado.

### Exploração e Limpeza de Dados

O processo de limpeza de dados incluiu a remoção de outliers, tratamento de dados ausentes e normalização das variáveis numéricas. Também foram realizadas análises exploratórias para entender a distribuição dos dados.

### Preparação dos Dados

Foram criadas features como médias móveis (de 10 e 30 dias) e o Índice de Força Relativa (RSI) para capturar tendências e padrões. Os dados foram divididos em conjuntos de treino e teste para avaliação do modelo.

## Modelagem e Otimização

### Modelos Utilizados

Testamos vários modelos de aprendizado de máquina, incluindo `MLPClassifier`, `RandomForest`, e `SVM`. Optamos pelo `MLPClassifier` por apresentar melhor desempenho em termos de acurácia e tempo de execução.

### Otimização de Hiperparâmetros

Utilizamos o `GridSearchCV` para otimizar os hiperparâmetros do modelo, buscando a melhor combinação de parâmetros para maximizar o desempenho.

### Validação do Modelo

Aplicamos validação cruzada K-Fold e calculamos métricas como acurácia, precisão, recall, F1-score, e AUC-ROC para avaliar o modelo.

## Implementação em Produção

### API Flask

Desenvolvemos uma API usando Flask para servir o modelo em produção e permitir previsões em tempo real. A API recebe dados de entrada, processa-os e retorna a previsão do modelo.

## Resultados e Discussão

### Desempenho do Modelo

O modelo apresentou uma acurácia de 84.30%, com precisão de 80.00% para a classe 'False' e 90.00% para a classe 'True', e recall de 90.34% para a classe 'False' e 78.61% para a classe 'True'. O F1-Score médio foi de 84.47%, indicando um bom desempenho geral.

### Visualizações de Dados

As visualizações incluem gráficos de evolução de preços, candlestick, volume de negociação, médias móveis, e correlação entre variáveis, ajudando a identificar padrões e tendências no mercado de Bitcoin.

### Limitações e Desafios

Entre as limitações estão a necessidade de re-treinamento periódico do modelo e a inclusão de novas features para melhorar a acurácia. Desafios incluem a alta volatilidade do mercado de criptomoedas.

## Conclusão e Próximos Passos

### Conclusões

O modelo mostrou-se eficaz na previsão de tendências do Bitcoin, mas há espaço para melhorias futuras.

### Próximos Passos

Sugestões de melhorias incluem a inclusão de novas features, re-treinamento periódico do modelo e refinamento das visualizações.

## Anexos

### Código Fonte

O código-fonte está disponível no GitHub: [IgaoWolf/Analise-Preditiva-de-Tendencias-do-Bitcoin-com-Mineracao-de-Dados](https://github.com/IgaoWolf/Analise-Preditiva-de-Tendencias-do-Bitcoin-com-Mineracao-de-Dados).

### Links Úteis

- **Dashboard**: Será necessário subir localmente.
- **Documentação**: [IgaoWolf/Analise-Preditiva-de-Tendencias-do-Bitcoin-com-Mineracao-de-Dados](https://github.com/IgaoWolf/Analise-Preditiva-de-Tendencias-do-Bitcoin-com-Mineracao-de-Dados).

## Como Executar a Aplicação

Siga os passos abaixo para configurar e executar a aplicação localmente:

### 1. Clone o Repositório

Clone este repositório para o seu ambiente local:

```bash
git clone https://github.com/IgaoWolf/Analise-Preditiva-de-Tendencias-do-Bitcoin-com-Mineracao-de-Dados.git
cd Analise-Preditiva-de-Tendencias-do-Bitcoin-com-Mineracao-de-Dados
```

### 2. Instale as Dependências

Crie um ambiente virtual (opcional, mas recomendado) e instale todas as dependências listadas em `requirements.txt`:

```bash
# Crie um ambiente virtual (opcional)
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate

# Instale as dependências
pip install -r requirements.txt
```

### 3. Execute a Aplicação

Execute o arquivo `app.py` para iniciar a aplicação:

```bash
python app/app.py
```

### 4. Acesse o Dashboard

Após iniciar a aplicação, acesse o dashboard no seguinte endereço:

- **Dashboard de Visualização**: [http://localhost:5000/dashboard](http://localhost:5000/dashboard)

## Estrutura de Gráficos e Indicadores

### Gráficos Disponíveis:

1. **Gráfico de Linha**: Mostra a evolução do preço de fechamento do Bitcoin ao longo do tempo.
2. **Gráfico de Candlestick**: Visualiza os movimentos de mercado, incluindo abertura, fechamento, alta e baixa.
3. **Heatmap de Correlação**: Mostra a correlação entre diferentes variáveis numéricas.
4. **Gráfico de Médias Móveis (SMA_10 e SMA_30)**: Indica tendências de mercado com base em médias móveis de curto e longo prazo.
5. **Gráfico de Volume de Negociação**: Mostra o volume de negociação do Bitcoin ao longo do tempo.
6. **Gráfico de RSI (Índice de Força Relativa)**: Indica condições de sobrecompra ou sobrevenda do ativo.
7. **Gráfico de Comparação de Preços de Abertura e Fechamento**: Mostra a diferença entre preços de abertura e fechamento.
8. **Gráfico de Regressão Linear**: Realiza uma previsão do preço usando regressão linear.

### Estratégias de Decisão de Compra

Para tomar uma decisão de compra informada, utilize a combinação de indicadores:

- **RSI (Índice de Força Relativa)**: Valores abaixo de 30 indicam uma oportunidade de compra.
- **Médias Móveis (SMA_10 e SMA_30)**: O cruzamento da média móvel de curto prazo acima da média de longo prazo sugere compra.
- **Gráfico de Candlestick**: Identifique padrões de reversão (ex.: Martelo).
- **Volume de Negociação**: Um aumento do volume com preço em alta confirma uma tendência de compra.
- **Regressão Linear**: Preço abaixo da linha de regressão sugere subvalorização.

## Contribuição

Sinta-se à vontade para enviar pull requests e abrir issues para sugerir melhorias ou relatar problemas. Toda contribuição é bem-vinda!

## Contato

Para dúvidas ou sugestões, entre em contato pelo e-mail: igaowolf@gmail.com.
---

Este `README.md` deve fornecer uma visão clara e completa do projeto, desde sua configuração e execução até o entendimento de como os gráficos e indicadores funcionam para apoiar decisões de investimento.
