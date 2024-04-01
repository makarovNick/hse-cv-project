FROM python:3.6-slim

ARG PORT=8501

EXPOSE $PORT
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    cmake \
    ninja-build \
    libc-dev \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip3 install -U --no-cache-dir pip -r docker_requirements.txt && pip3 install -e .

RUN pip install poetry

EXPOSE PORT

CMD ["streamlit run app.py --server.port $PORT"]
