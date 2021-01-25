FROM python:3.9

WORKDIR /workspace
COPY Pipfile Pipfile.lock cucumber-format.patch setup.py /workspace/
RUN \
  pip install pipenv && \
  pipenv install --dev --system
RUN \
  patch -d /usr/local/lib/python3.9/site-packages/behave/formatter -p1 < cucumber-format.patch

ENV PYTHONDONTWRITEBYTECODE=1
