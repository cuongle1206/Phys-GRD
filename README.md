
# Official implementation of PhysGRD

Physics-informed Ground Reaction Dynamics from Human Motion Capture

<div align="center">
<img src="animations/Teaser.png" width="1000" alt="logo"/>
</div>

[![Paper](https://img.shields.io/badge/arXiv-2507.01340-red)](https://arxiv.org/abs/2507.01340) [![IEEE](https://img.shields.io/badge/IEEE-HSI_2025-blue?logo=openreview)](https://ieeexplore.ieee.org/document/11142382)

## Installation

Install conda enviroment.

```
conda env create -f environment.yml
```

## Experiments

To preprocess the data:

```
python ./GRF/scripts/preprocess.py
```

To train the model:

```
python main.py
```

This willl automaticall create the train, validate and test sets.

To test the mode:

```
python test.py
```

To visuallize a sample:

```
cd Visualization

python visualize_target_pred_s7.py

python viz_better.py
```

If you find our work helpful, please cite the paper as
```bibtex
@inproceedings{le2025_physgrd,
  title     = {Physics-informed Ground Reaction Dynamics from Human Motion Capture},
  author    = {Le, Cuong and Le, Huy-Phuong, Le, Duc and Duong, Minh-Thien and Nguyen, Van-Binh and Le, My-Ha},
  booktitle = {Conference on Human System Interactions (HSI)},
  year      = {2025}
}
```

Acknowledgements

The repository is based on the [GroundLink](https://github.com/hanxingjian/GroundLink) dataset, and many thanks to the authors of GroundLink.
