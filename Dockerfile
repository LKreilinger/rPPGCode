# Zugrundeliegendes Image
#FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime
FROM wallies/python-cuda:3.8-cuda11.3-runtime



# Helpers
ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# activate venv
#ENV VIRTUAL_ENV=/opt/venv
#RUN python3 -m venv $VIRTUAL_ENV
#ENV PATH="$VIRTUAL_ENV/bin:$PATH"

ARG USER=lkreilinger
ARG UID=1004
ARG GID=1004
# default password for user
ARG PW=29653847
# Option1: Using unencrypted password/ specifying password
RUN useradd -m ${USER} --uid=${UID} && echo "${USER}:${PW}" | \
      chpasswd

# Option2: Using the same encrypted password as host
#COPY /etc/group /etc/group
#COPY /etc/passwd /etc/passwd
#COPY /etc/shadow /etc/shadow

# install the cv2 dependencies
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

#RUN wandb login b2b87b7a47a54d74a79cf8ceb131c26efe9418a5
USER ${UID}:${GID}
WORKDIR /
COPY . ./

# Install dependencies:
RUN pip install --no-cache-dir -r requirements.txt --user
RUN pip install --no-cache-dir torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html --user

CMD ["python3", "__main__.py"]