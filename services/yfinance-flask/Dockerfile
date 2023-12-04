FROM python:3.8-slim

RUN useradd -ms /bin/bash appuser

RUN pip install --upgrade pip && \
    pip install flask requests yfinance pandas && \
    rm -rf /root/.cache

WORKDIR /app
COPY main.py /app
COPY storage.py /app
#chown -R appuser:appuser /app

USER appuser

EXPOSE 5000

CMD ["python", "main.py"]

