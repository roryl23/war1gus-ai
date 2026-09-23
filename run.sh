#!/usr/bin/env bash

package_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

if [[ -z ${WAR1GUS_AI_LOG_PATH:-} ]]; then
    WAR1GUS_AI_LOG_PATH="$package_root/War1gusAI.log"
fi
export WAR1GUS_AI_LOG_PATH

if [[ ! -v OPENBLAS_NUM_THREADS ]]; then
    OPENBLAS_NUM_THREADS=1
fi
export OPENBLAS_NUM_THREADS

exec julia --threads=2 --optimize=3 --interactive --project="$package_root" -e \
'using War1gusAI; exit(War1gusAI.julia_main())' -- "$@"
