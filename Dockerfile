# Zugrundeliegendes Image
#FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime
FROM wallies/python-cuda:3.8-cuda11.3-runtime



# Helpers
ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# activate venv
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

ARG USER=lkreilinger
ARG UID=1004
ARG GID=1004
# default password for user
ARG PW=29653847

# Option2: Using the same encrypted password as host
COPY /etc/group /etc/group
COPY /etc/passwd /etc/passwd
COPY /etc/shadow /etc/shadow
USER ${UID}:${GID}
WORKDIR /home/${USER}
#RUN mkdir -p /workdir
#RUN chown 1004:1004 /workdir
#USER 1004:1004
#WORKDIR /workdir
#WORKDIR /home/lkreilinger/Masterarbeit
# Install dependencies:
WORKDIR /
COPY . ./
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

CMD ["python3", "__main__.py"]