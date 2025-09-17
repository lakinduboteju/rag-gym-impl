#!/bin/bash

docker run -d -t --name rag-gym \
    -v ./:/app \
    -w /app \
    -p 6678:5678 \
    rag-gym:latest bash
