# Use uma imagem base do Python
FROM python:3.11-slim

# Defina o diretório de trabalho no container
WORKDIR /app

# Copie todos os arquivos do diretório local para o diretório de trabalho no container
COPY . /app

# Instale as dependências listadas no requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Exponha a porta que o servidor irá rodar
EXPOSE 5000

# Comando para iniciar a aplicação
CMD ["python", "app.py"]
