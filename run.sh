#!/usr/bin/env bash

julia --threads=2 --optimize=3 --interactive --project -e \
'using War1gusAI; War1gusAI.real_main();'
