FROM python:3.8-slim

RUN useradd -ms /bin/bash appuser

RUN pip install --upgrade pip && \
    pip install flask flask-cors redis requests && \
    rm -rf /root/.cache

WORKDIR /app
COPY main.py /app
COPY backend.py /app

USER appuser

EXPOSE 5000

CMD ["python", "main.py"]

