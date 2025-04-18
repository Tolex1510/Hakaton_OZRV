FROM python:3.13-alpine3.21

WORKDIR app

COPY . .
COPY ../Neuro ./Neuro

RUN pip install -r --no-cache-dir requirements.txt

CMD ["python", "app.py"]