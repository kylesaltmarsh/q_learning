FROM python:3.6

RUN ["mkdir", "frozen_lake"]
WORKDIR "frozen_lake"

COPY "requirements.txt" .
RUN ["pip", "install", "-r", "requirements.txt"]

COPY "run.sh" .
COPY "frozen_lake.py" .

RUN ["chmod", "+x", "./run.sh"]
ENTRYPOINT ["./run.sh"]