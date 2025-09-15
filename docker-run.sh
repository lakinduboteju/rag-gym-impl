#!/bin/bash

docker run -d -t --name rag-gym \
    -v ./:/app \
    -w /app \
    rag-gym:latest bash
