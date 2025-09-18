FROM python:3.12-slim

WORKDIR /app

# Instala dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copia los archivos del proyecto
COPY . .

# Copia explícitamente la carpeta de modelos (si existe)
COPY models/* ./models/

RUN ls -la ./models/

# Instala las dependencias de Python
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Expone el puerto para Streamlit
EXPOSE 8501

# Comando por defecto: bash (puedes sobreescribirlo en docker run)
CMD ["bash"]