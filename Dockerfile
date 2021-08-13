FROM python:3.6

WORKDIR /ARLAffPoseDatasetUtils

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY ./src ./src
