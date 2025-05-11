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

RUN pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu118
RUN pip install pytorch-lightning

RUN pip install transformers==4.43.2
RUN pip install matplotlib
RUN pip install scikit-learn
RUN pip install peft
RUN pip install traker

CMD ["python3"]