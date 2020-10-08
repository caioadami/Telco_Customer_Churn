FROM python:3.8-slim
COPY . /app
WORKDIR /app
#RUN apt-get update && \
 #   apt-get install -y --no-install-recommends \
  #      ca-certificates \
   #     cmake \
    #    build-essential \
     #   gcc \
      #  g++ 
RUN pip install -r requirements.txt
#RUN python main.py
EXPOSE 5000
CMD python ./main.py