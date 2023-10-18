# Usa la imagen base de Python
FROM python:3.9-slim

# Establece el directorio de trabajo
WORKDIR /app

# Copia los archivos locales al contenedor
COPY app.py requirements.txt Procfile /app/

RUN pip install --no-cache-dir numpy==1.19.5

# Instala las dependencias desde requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expone el puerto en el que el servidor Flask va a ejecutarse
EXPOSE 5000

# Define el comando para ejecutar la aplicación
CMD ["python", "app.py"]