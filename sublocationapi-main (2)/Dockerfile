FROM python:3.8-slim

WORKDIR /usr/src/sublocation_api_cloud
COPY . .
RUN apt-get update \
     && apt-get install -y \
        libgl1-mesa-glx \
        libx11-xcb1 \
        ffmpeg \
        libsm6 \
        libxext6 \        
     && apt-get clean all \
     && rm -r /var/lib/apt/lists/*

RUN /usr/local/bin/python -m pip install --upgrade pip
RUN --mount=type=cache,target=/root/.cache/pip pip3 install -r requirements.txt
ADD . /usr/src/sublocation_api_cloud
CMD [ "python", "./app.py" ]
EXPOSE 5000