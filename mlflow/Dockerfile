# Defining base image
FROM python:3.8.2-slim
# Installing MLflow from PyPi
RUN pip install mlflow[extras]==1.9.1
# Defining start up command
EXPOSE 5000
ENTRYPOINT ["mlflow", "server"]