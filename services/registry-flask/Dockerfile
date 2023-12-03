FROM python:3.8-slim

RUN useradd -ms /bin/bash appuser

RUN pip install --upgrade pip && \
    pip install flask requests scikit-learn pandas tensorflow keras && \
    rm -rf /root/.cache

WORKDIR /app
COPY main.py /app
COPY file_storage.py /app
RUN mkdir -p /app/models && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 5000

CMD ["python", "main.py"]

