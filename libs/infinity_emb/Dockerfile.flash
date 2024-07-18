FROM nvcr.io/nvidia/pytorch:24.06-py3


WORKDIR /app

RUN apt-get update && apt-get install build-essential python3-dev python3.10-venv python3.10 curl -y 

RUN python -m venv .venv
RUN source .venv/bin/activate
RUN pip install flash_attn

COPY test.py /app


RUN pip install infinity-emb[all]

RUN infinity_emb v2 --model-id dunzhang/stella_en_1.5B_v5 --engine torch --preload-only || [ $? -eq 3 ]
ENTRYPOINT ["infinity_emb"]