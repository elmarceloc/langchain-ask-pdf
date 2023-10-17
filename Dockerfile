# Usa la imagen base de Python
FROM python:3.8-slim

# Establece el directorio de trabajo
WORKDIR /app

# Copia los archivos locales al contenedor
COPY app.py /app/

# Instala las dependencias
RUN pip install Flask

# Expone el puerto en el que el servidor Flask va a ejecutarse
EXPOSE 5000

# Define el comando para ejecutar la aplicación
CMD ["python", "app.py"]