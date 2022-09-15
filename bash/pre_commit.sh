#!/usr/bin/env bash
set -euxo pipefail

find . -type f -exec dos2unix {} \;
