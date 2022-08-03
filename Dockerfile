# Zugrundeliegendes Image
#FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime
FROM wallies/python-cuda:3.8-cuda11.3-runtime

# Information
LABEL version="0.0.1"
LABEL maintainer="Laurens Kreilinger"
#LABEL org.opencontainers.image.source = "https://github.com/SauravMaheshkar/gnn-lspe"


# Kopieren des aktuellen Verzeichnisses in /laurens
#ADD /laurens

# Helpers
# PYTHONUNBUFFERED: python output direkt in logs -> real time
# DEBIAN_FRONTEND: avoid debconf warnings while building
ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# activate venv
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install dependencies:
WORKDIR /
COPY . ./
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

CMD ["python3", "__main.py"]