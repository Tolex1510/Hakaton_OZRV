FROM python:3.13-alpine3.21

WORKDIR backend

COPY . .
COPY ../Neuro ./Neuro

RUN pip install -r requirements.txt

EXPOSE 6080
CMD ["python", "app.py"]