FROM python:3.6.5

WORKDIR /app/
ADD requirements.txt .
RUN pip install -r requirements.txt

ADD . .

EXPOSE 5000
CMD gunicorn -b 0.0.0.0:5000 wsgi
