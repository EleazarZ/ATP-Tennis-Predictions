FROM python:3.8

COPY app/flaskapp/predict_api /app/flaskapp/predict_api
COPY app/flaskapp/run.py /app/flaskapp/
COPY models/ /models/

WORKDIR /app

COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt 

EXPOSE 5000

CMD ["python3", "flaskapp/run.py"]