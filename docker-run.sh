#!/bin/bash

docker run -d -t --name rag-gym \
    -v ./:/app \
    -w /app \
    -p 6678:5678 \
    --network docker_ragflow \
    rag-gym:latest bash
