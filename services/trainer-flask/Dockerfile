FROM python:3.8-slim

RUN pip install --upgrade pip && \
    pip install flask requests scikit-learn pandas

WORKDIR /app
COPY main.py /app
COPY training_module.py /app
COPY processor_rules.py /app

EXPOSE 5000

CMD ["python", "main.py"]
