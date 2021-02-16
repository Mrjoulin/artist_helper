FROM python:3.6.5

WORKDIR /app/
ADD requirements.txt .
RUN apt-get update && apt-get install -y libsm6 libxext6 libxrender-dev python3-tk
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

ADD . .

EXPOSE 8000
# Run tests
# Run production
CMD python3 src/test.py && gunicorn --threads 10 -b 0.0.0.0:8000 wsgi
