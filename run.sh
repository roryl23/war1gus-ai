#!/usr/bin/env bash

julia --threads=2 --interactive --project -e \
'using war1gus; war1gus.real_main();'
