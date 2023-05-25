FROM python:3.8

ENV API_ENDPOINT="http://127.0.0.1:5000/predict/"

COPY app/dashapp/app.py /app/dashapp/
COPY app/dashapp/views.py /app/dashapp/
COPY models/config.pkl /models/

WORKDIR /app

COPY requirements.txt ./

RUN pip3 install --no-cache-dir -r requirements.txt 

EXPOSE 8050

CMD ["python3", "dashapp/app.py"]