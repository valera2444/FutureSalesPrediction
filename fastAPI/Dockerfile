
FROM python:3.12-slim


WORKDIR /code


COPY ./requirements.txt /code/requirements.txt


RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt


COPY ./app /code/app

#Lightbm error if omit this (libgomp1 - for multiprocessing)

RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
RUN apt-get -y install curl
RUN apt-get install libgomp1



CMD ["fastapi", "run", "app/main.py", "--port", "5050"]