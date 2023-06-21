#!/usr/bin/env bash
set -euxo pipefail

#find . -type f -exec dos2unix {} \;
find . -type f \( -not -path "./.git/*" -a -not -path "./.idea/*" -a -not -path "./__pycache__/*" \) -exec dos2unix {} \;

#files='lazaro/ utils/ scripts/ *.py'

files=(scripts/ utils)
isort -rc "${files[@]}"
black "${files[@]}"
flake8 "${files[@]}"
mypy "${files[@]}"