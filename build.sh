#!/bin/bash

julia --threads=auto --project compile.jl --incremental && \
chmod +x build/bin/war1gus-ai
