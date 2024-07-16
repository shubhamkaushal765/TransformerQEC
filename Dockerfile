FROM pytorch/pytorch:latest

# copy requirements.txt into the docker image
COPY requirements.txt /

RUN pip install -r /requirements.txt

WORKDIR /TransformerQEC

# docker build -t shubhamkaushal765/transformerqec:latest .

# when running on Microsoft Windows -> Git Bash
# MSYS_NO_PATHCONV=1 docker run --rm -it -v $(pwd):/TransformerQEC shubhamkaushal765/transformerqec:latest

# when running on Linux
# docker run --rm -it -v $(pwd):/TransformerQEC shubhamkaushal765/transformerqec:latest