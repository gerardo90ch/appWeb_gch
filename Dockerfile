FROM ubuntu:20.04
RUN apt-get update && \
    apt-get install -y python3.9 && \ 
    apt-get install -y python3-pip
COPY . /app
WORKDIR /app
RUN python3.9 -m pip install -r requirements.txt
EXPOSE 8000
EXPOSE 5000
EXPOSE 5001
CMD python3 -m http.server & python3.9 app.py & python3.9 baseDatos.py
