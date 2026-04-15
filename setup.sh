#!/bin/bash
# All dependencies including spaCy models are declared in requirements.txt.
# The en_core_web_sm and en_core_sci_sm wheels are installed automatically
# by pip — do not run `python -m spacy download` separately as it will
# bypass the pinned versions and install whatever the CDN currently serves.

set -e

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
