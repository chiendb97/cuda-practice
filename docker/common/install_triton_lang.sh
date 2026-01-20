

#!/bin/bash

set -ex

pip3 install triton==3.5.1

# Clean up pip cache and temporary files
pip3 cache purge
rm -rf ~/.cache/pip
rm -rf /tmp/*
