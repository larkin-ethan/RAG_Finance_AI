#!/usr/bin/env bash
# Delete the cluster. Data (Postgres, caches, backups) stays in $FINRAG_STORAGE
# on the host, so `up.sh` brings everything back.
set -euo pipefail
k3d cluster delete finrag
echo "Cluster deleted. Data kept in ${FINRAG_STORAGE:-$HOME/.finrag/storage}."
echo "To erase it permanently: rm -rf ${FINRAG_STORAGE:-$HOME/.finrag/storage}  (irreversible)"
