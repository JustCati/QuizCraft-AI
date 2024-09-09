FROM pytorch/pytorch:latest
RUN apt-get update && apt-get install -y git poppler-utils

COPY requirements.txt /workspace/requirements.txt
RUN python3 -m pip install -r /workspace/requirements.txt
RUN pip install -U torch torchvision torchaudio
RUN apt install -y software-properties-common
RUN apt-add-repository -y ppa:fish-shell/release-3
RUN apt update
RUN apt install -y fish

RUN useradd -ms /bin/fish cati
USER cati
WORKDIR /workspace
