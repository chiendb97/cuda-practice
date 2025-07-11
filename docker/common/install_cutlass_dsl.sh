#!/bin/bash

set -ex

pip3 install nvidia-cutlass-dsl==4.1.0.dev0

# Clean up pip cache and temporary files
pip3 cache purge
rm -rf ~/.cache/pip
rm -rf /tmp/*