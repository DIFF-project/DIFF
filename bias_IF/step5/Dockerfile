FROM harbor.lins.lab/determinedai/environments:cuda-11.8-pytorch-2.0-gpu-mpi-0.31.1

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    vim \
    build-essential \
    screen \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu118

RUN pip install -r requirements.txt -i https://mirrors.bfsu.edu.cn/pypi/web/simple
CMD ["python3"]