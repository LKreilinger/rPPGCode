# Zugrundeliegendes Image
# The torch wheel from requirements.txt brings its own CUDA libraries
FROM python:3.12-slim



# Helpers
ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# activate venv
#ENV VIRTUAL_ENV=/opt/venv
#RUN python3 -m venv $VIRTUAL_ENV
#ENV PATH="$VIRTUAL_ENV/bin:$PATH"

ARG USER=xxx
ARG UID=1004
ARG GID=1004
# default password for user
ARG PW=123456789
# Option1: Using unencrypted password/ specifying password
RUN useradd -m ${USER} --uid=${UID} && echo "${USER}:${PW}" | \
      chpasswd

# Option2: Using the same encrypted password as host
#COPY /etc/group /etc/group
#COPY /etc/passwd /etc/passwd
#COPY /etc/shadow /etc/shadow

# install the cv2 dependencies
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 libgl1 libglib2.0-0 -y


USER ${UID}:${GID}
WORKDIR /
COPY . ./

# Install dependencies:
RUN pip install --no-cache-dir -r requirements.txt --user
CMD ["python3", "__main__.py", "--docker"]
