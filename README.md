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

## Citation and Dataset

The creation of the Transitions in Parenting of Teens (TPOT) dataset was funded by NIH grant #5R01 HD081362 (awarded to Lisa B. Sheeber and Nicholas B. Allen). When referring to the TPOT dataset, please cite
```bibtex
@article{nelson2021psychobiological,
  title={Psychobiological markers of allostatic load in depressed and nondepressed mothers and their adolescent offspring},
  author={Nelson, Benjamin W and Sheeber, Lisa and Pfeifer, Jennifer and Allen, Nicholas B},
  journal={Journal of Child Psychology and Psychiatry},
  volume={62},
  number={2},
  pages={199--211},
  year={2021},
  publisher={Wiley Online Library}
}
```

When using the [multimodal features and annotations](https://cmu.box.com/s/o2lvyd2jc0c72dreq0w3bvdg9wg6g309) collected as part of this ICMI paper, please cite

```bibtex
@inproceedings{wortwein2021human,
  title={Human-Guided Modality Informativeness for Affective States},
  author={W{\"o}rtwein, Torsten and Sheeber, Lisa B and Allen, Nicholas and Cohn, Jeffrey F and Morency, Louis-Philippe},
  booktitle={Proceedings of the 2021 International Conference on Multimodal Interaction},
  pages={728--734},
  year={2021}
}
```
