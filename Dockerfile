FROM python:3.6.5

WORKDIR /app/
ADD requirements.txt .
RUN pip install -r requirements.txt

ADD . .
EXPOSE 5000

CMD ["python3", "wsgi.py"]
