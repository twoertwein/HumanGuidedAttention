# HumanGuidedAttention
The code for ["Human-Guided Modality Informativeness for Affective States"](https://dl.acm.org/doi/abs/10.1145/3462244.3481004) at ICMI 2021.

## Installation
Python 3.9 and [poetry](https://github.com/python-poetry/poetry) are required.
```sh
poetry add git+https://github.com:twoertwein/HumanGuidedAttention.git
poetry run pip install torchvision==0.11.1+cu102 -f https://download.pytorch.org/whl/torch_stable.html
```

To locate `human_guided_attention.train`, please use 
```sh
poetry run python -c "from human_guided_attention import train; print(train.__file__)"
```

## Usage
Run the grid-search for the guided model:
```sh
poetry run python path/to/train.py --GUIDED
```
