FROM python:3.11.3

RUN apt update
RUN pip install tensorflow
RUN pip install Pillow

WORKDIR /usr/app/src

COPY app /app
COPY mnt /mnt

CMD ["echo", "Docker is ready"]