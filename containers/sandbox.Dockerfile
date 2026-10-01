FROM python:3.11-slim
RUN pip install --no-cache-dir --no-deps evalplus==0.3.1 numpy==2.2.6 psutil==7.0.0
COPY src/transferlab/codec.py /opt/transferlab/codec.py
USER 65534:65534
