# Usa la imagen base de Python
FROM python:3.8.18-bookworm


# Establece el directorio de trabajo
WORKDIR /app

# Copia los archivos locales al contenedor
COPY app.py requirements.txt Procfile /app/

RUN pip3 install --upgrade -i https://mirrors.aliyun.com/pypi/simple pip
# Instala las dependencias desde requirements.txt
#RUN pip install --no-cache-dir numpy==1.19.5
RUN pip install --no-cache-dir -r requirements.txt


# Expone el puerto en el que el servidor Flask va a ejecutarse
EXPOSE 8080

# Define el comando para ejecutar la aplicación
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8080", "app:app"]