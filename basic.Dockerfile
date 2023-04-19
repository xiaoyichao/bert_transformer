FROM docker.haohaozhu.me/hhz-ml/python:basic
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt