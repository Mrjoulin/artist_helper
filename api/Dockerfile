FROM python:3.6.5

WORKDIR /app/
ADD requirements.txt .
RUN pip install -r requirements.txt

ADD . .

EXPOSE 8000
CMD gunicorn --threads 10 -b 0.0.0.0:8000 wsgi
