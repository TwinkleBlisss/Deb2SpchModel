# Backend with speech separation model for the Deb2Spch application.

Python used in dev: `3.12.10`

## Run
`$ python app.py`

## Tests
`$ pytest -v`

## Before run in prod!!!
In requirements.txt replace: \
`torch==2.7.0+cpu` \
`torchaudio==2.7.0+cpu` \
With: \
`torch==2.7.0` \
`torchaudio==2.7.0` \

The CPU version was used because an error may occur without the GPU.
