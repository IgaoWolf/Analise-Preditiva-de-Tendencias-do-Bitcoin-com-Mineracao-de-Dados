# Use uma imagem base oficial do Python
FROM python:3.11-slim

# Defina o diretório de trabalho no contêiner
WORKDIR /app

# Copie o arquivo requirements.txt para o diretório de trabalho
COPY requirements.txt .

# Instale as dependências
RUN pip install --no-cache-dir -r requirements.txt

# Copie todos os arquivos do projeto para o contêiner
COPY . .

# Exponha a porta que o Flask vai rodar
EXPOSE 5000

# Comando para executar o aplicativo
CMD ["python", "app/app.py"]
