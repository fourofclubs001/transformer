# Use the slim version of Python 3
FROM python:3.10.9-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget nano && \
    rm -rf /var/lib/apt/lists/*
RUN apt-get upgrade -y

RUN pip install jupyter

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES all

ENV CUDA_LAUNCH_BLOCKING 1

RUN mkdir /workspace
RUN echo "umask 002" >> /etc/profile
RUN chmod -R 777 /workspace

WORKDIR /workspace

CMD ["jupyter", "notebook", "--NotebookApp.token=''", "--NotebookApp.password=''", "--ip=0.0.0.0", "--allow-root"]

EXPOSE 8888