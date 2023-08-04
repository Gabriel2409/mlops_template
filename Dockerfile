ARG BASE_IMAGE=python:3.10
FROM $BASE_IMAGE

# install project requirements
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache -r /tmp/requirements.txt && rm -f /tmp/requirements.txt
