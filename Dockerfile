FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    vim \
    build-essential \
    screen \
    && rm -rf /var/lib/apt/lists/*

# RUN pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121

RUN pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu118
RUN pip install pytorch-lightning

RUN pip install transformers==4.43.2 -i https://mirrors.bfsu.edu.cn/pypi/web/simple
RUN pip install matplotlib -i https://mirrors.bfsu.edu.cn/pypi/web/simple
RUN pip install scikit-learn -i https://mirrors.bfsu.edu.cn/pypi/web/simple ninja
RUN pip install peft -i https://mirrors.bfsu.edu.cn/pypi/web/simple
RUN pip install traker -i https://mirrors.bfsu.edu.cn/pypi/web/simple

CMD ["python3"]