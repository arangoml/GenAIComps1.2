# Retriever Microservice with ArangoDB (work-in-progress)

## 🚀Start Microservice with Python

### Install Requirements

```bash
pip install -r requirements.txt
```

### Start ArangoDB Server

To launch ArangoDB locally, first ensure you have docker installed. Then, you can launch the database with the following docker command.


```bash
docker run -d   --name arango-vector-db  -p 8529:8529   -e ARANGO_ROOT_PASSWORD=password   arangodb/arangodb:3.12.4   --experimental-vector-index=true
```

### Setup Environment Variables

```bash
export no_proxy=${your_no_proxy}
export http_proxy=${your_http_proxy}
export https_proxy=${your_http_proxy}
export ARANGO_URL="http://localhost:8529"
export ARANGODB_USERNAME=root
export ARANGODB_PASSWORD=test
export ARANGO_DB_NAME=_system

```


## 🚀Start Microservice with Docker

### Build Docker Image

```bash
cd ../../
docker build -t opea/retriever:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/retrievers/src/Dockerfile .
```


### With the Docker compose

First, set the env variables:

```bash

model=BAAI/bge-base-en-v1.5
export host_ip="127.0.0.1"
export TEI_EMBEDDING_ENDPOINT="http://${host_ip}:6060"  # if you want to use the hosted embedding service, example: "http://127.0.0.1:6060"
```

Text embeddings inference service expects the `RETRIEVE_MODEL_ID` variable to be set.

```bash
export EMBEDDING_MODEL_ID=BAAI/bge-base-en-v1.5
```

Note that following docker compose sets the `network_mode: host` in retriever image to allow local vector store connection.
This will start the both the embedding and retriever services:

```bash
cd ../deployment/docker_compose
export service_name="retriever-arango"
docker compose -f compose.yaml up ${service_name} -d
```


## 🚀3. Consume Retriever Service

### 3.1 Check Service Status

```bash
curl http://${your_ip}:7000/v1/health_check \
  -X GET \
  -H 'Content-Type: application/json'
```

### 3.2 Consume Embedding Service

To consume the Retriever Microservice, you can generate a mock embedding vector of length 768 with Python.

```bash
export your_embedding=$(python -c "import random; embedding = [random.uniform(-1, 1) for _ in range(768)]; print(embedding)")
curl http://${your_ip}:7000/v1/retrieval \
  -X POST \
  -d "{\"text\":\"What is the revenue of Nike in 2023?\",\"embedding\":${your_embedding}}" \
  -H 'Content-Type: application/json'
```

```bash
export your_embedding=$(python -c "import random; embedding = [random.uniform(-1, 1) for _ in range(768)]; print(embedding)")
curl http://${your_ip}:7000/v1/retrieval \
  -X POST \
  -d "{\"text\":\"What is the revenue of Nike in 2023?\",\"embedding\":${your_embedding},\"search_type\":\"similarity\", \"k\":4}" \
  -H 'Content-Type: application/json'
```

```bash
export your_embedding=$(python -c "import random; embedding = [random.uniform(-1, 1) for _ in range(768)]; print(embedding)")
curl http://${your_ip}:7000/v1/retrieval \
  -X POST \
  -d "{\"text\":\"What is the revenue of Nike in 2023?\",\"embedding\":${your_embedding},\"search_type\":\"similarity_distance_threshold\", \"k\":4, \"distance_threshold\":1.0}" \
  -H 'Content-Type: application/json'
```

```bash
export your_embedding=$(python -c "import random; embedding = [random.uniform(-1, 1) for _ in range(768)]; print(embedding)")
curl http://${your_ip}:7000/v1/retrieval \
  -X POST \
  -d "{\"text\":\"What is the revenue of Nike in 2023?\",\"embedding\":${your_embedding},\"search_type\":\"similarity_score_threshold\", \"k\":4, \"score_threshold\":0.2}" \
  -H 'Content-Type: application/json'
```


```bash
export your_embedding=$(python -c "import random; embedding = [random.uniform(-1, 1) for _ in range(768)]; print(embedding)")
curl http://${your_ip}:7000/v1/retrieval \
  -X POST \
  -d "{\"text\":\"What is the revenue of Nike in 2023?\",\"embedding\":${your_embedding},\"search_type\":\"mmr\", \"k\":4, \"fetch_k\":20, \"lambda_mult\":0.5}" \
  -H 'Content-Type: application/json'
```


