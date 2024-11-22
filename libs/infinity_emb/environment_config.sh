#!/bin/bash
#export OTEL_TRACES_SAMPLER=parentbased_always_off
export OTEL_RESOURCE_ATTRIBUTES=service.name=${SHERLOCK_SERVICE_NAME},host.name=${POD_NAME},host.ip=${POD_IP}
export OTEL_EXPORTER_OTLP_ENDPOINT=http://${HOST_IP}:5680
infinity_emb v2 --model-id /models --dtype ${DTYPE}
