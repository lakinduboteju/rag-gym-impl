FROM python:3.12-slim-bookworm

ENV POETRY_HOME=/etc/poetry
ENV PATH="$PATH:$POETRY_HOME/bin:$POETRY_HOME/venv/bin:/root/.local/bin"

RUN apt update && apt install -y curl

RUN curl -sSL https://install.python-poetry.org |  python -

