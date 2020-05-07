FROM python:3.6

# set the working directory
RUN ["mkdir", "frozen_lake"]
WORKDIR "frozen_lake"

# install code dependencies
COPY "requirements.txt" .
RUN ["pip", "install", "-r", "requirements.txt"]

# install environment dependencies
COPY "run.sh" .
COPY "frozen_lake.py" .

# provision environment
# ENV FLASK_APP app.py
RUN ["chmod", "+x", "./run.sh"]
# EXPOSE 8080
ENTRYPOINT ["./run.sh"]
CMD ["train"]