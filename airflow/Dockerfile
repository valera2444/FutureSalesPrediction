FROM apache/airflow:2.10.3-python3.12

ADD requirements.txt .


RUN python -m pip install --upgrade pip

RUN pip install --no-cache-dir -r ./requirements.txt

USER root

COPY future_sales_prediction ./future_sales_prediction

#Lightbm error if omit this (libgomp1 - for multiprocessing)

RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
RUN apt-get -y install curl
RUN apt-get install libgomp1


#MLFlow gives error without this (or not))))
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git