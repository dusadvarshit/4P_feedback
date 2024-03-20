#FROM python:3.6
FROM ubuntu:18.04
RUN apt-get update
RUN apt install -y ffmpeg
RUN apt install -y python3.6
RUN apt-get install -y python3-pip
ADD requirements.txt /
RUN pip3 install -r requirements.txt
