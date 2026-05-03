FROM python:3.11-slim

WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r /workspace/requirements.txt

COPY . /workspace

RUN chmod +x /workspace/docker/start_all.sh

EXPOSE 5173 8000

CMD ["/workspace/docker/start_all.sh"]
