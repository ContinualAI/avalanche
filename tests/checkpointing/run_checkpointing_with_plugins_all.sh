#!/usr/bin/env bash
# Script used to manually test various combinations of plugins when used with
# the checkpointing functionality.
set -euo pipefail
./test_checkpointing_plugins.sh
./test_checkpointing_plugins.sh replay
./test_checkpointing_plugins.sh cwr
./test_checkpointing_plugins.sh si
./test_checkpointing_plugins.sh ewc
./test_checkpointing_plugins.sh lwf
./test_checkpointing_plugins.sh gdumb
./test_checkpointing_plugins.sh replay cwr
./test_checkpointing_plugins.sh replay si
./test_checkpointing_plugins.sh replay ewc
./test_checkpointing_plugins.sh replay lwf
