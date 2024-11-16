FROM apache/airflow:2.10.3-python3.12

ADD requirements.txt .


RUN python -m pip install --upgrade pip

RUN pip install --no-cache-dir "apache-airflow==2.10.3" -r ./requirements.txt

USER root
COPY data ./data 

#for Path(destination_path).mkdir(parents=True, exist_ok=True) in etl.py
RUN chown -R 1000:0 ./data



COPY future_sales_prediction ./future_sales_prediction

#Lightbm error if omit this

RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
RUN apt-get -y install curl
RUN apt-get install libgomp1


#MLFlow gives error without this (or not))))
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git