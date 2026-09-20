#!/bin/bash

julia --project -e 'using Pkg; Pkg.instantiate()' && \
julia --threads=auto --optimize=3 --project compile.jl --incremental && \
mv build/bin/War1gusAI build/bin/War1gusAI.bin && \
install -m 755 launcher.sh build/bin/War1gusAI
