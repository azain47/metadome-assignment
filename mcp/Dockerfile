FROM python:3.12-slim

WORKDIR /srv
COPY . /srv/mcp
RUN pip install --no-cache-dir /srv/mcp "mcp[cli]==1.30.0" "boto3==1.44.3"

EXPOSE 8000
CMD ["blender-mcp", "--transport", "http", "--host", "0.0.0.0", "--port", "8000"]
