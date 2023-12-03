FROM python:3.8-slim

RUN pip install --upgrade pip && \
    pip install flask requests scikit-learn pandas tensorflow keras

WORKDIR /app
COPY main.py /app
COPY lrnstak-*.whl /app/
RUN pip install ./lrnstak-*.whl

EXPOSE 5000

CMD ["python", "main.py"]
