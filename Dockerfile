FROM python:3.9-slim

WORKDIR /app


COPY app.py requirements.txt Procfile /app/

RUN pip freeze > requirements.txt

# Instala las dependencias desde requirements.txt
RUN pip install --no-cache-dir numpy==1.19.5

EXPOSE 5000

# Define el comando para ejecutar la aplicación
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]