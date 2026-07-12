# A Time-Aware Self-Supervised Framework for Anomaly Detection in Temporal Graphs
This is the code for **[A Time-Aware Self-Supervised Framework for Anomaly Detection in Temporal Graphs](https://github.com/slitiWassim/NFT-Suspicious-Activity)** .

[![License](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.4](https://img.shields.io/badge/PyTorch-1.13-orange.svg)](https://pytorch.org/)
[![PyTorch Geometric 2.6](https://img.shields.io/badge/PyG-2.6-purple.svg)](https://pytorch-geometric.readthedocs.io/en/2.7.0/install/installation.html)

### [🌐 Project](https://slitiwassim.github.io/T-ADTG/) | [📄 Paper]()


<p align="center">
  <a href="static/images/paper_.png" target="_blank">
    <img
      src="static/images/paper_.png"
      width="90%"
      style="border: 2px solid rgb(201, 196, 196);"
    >
  </a>
</p>



## Setup
The code can be run under any environment with Python 3.9.25 and above.
(It may run with lower versions, but we have not tested it).


Clone this repo:

    git clone https://github.com/slitiWassim/T-ADTG.git
    cd T-ADTG/


Install the required packages:

    pip install -r requirements.txt
  

## Framework Overview

<p align="center">
  <a href="static/images/T-ADTG.png" target="_blank">
    <img
      src="static/images/T-ADTG.png"
      width="100%"
      style="border: 2px solid rgb(201, 196, 196);"
    >
  </a>
</p>


We evaluate `T-ADTG` on:
| Dataset | Link                                                                                  |
|--|---------------------------------------------------------------------------------------|
| Wikipedia | [![Google drive](https://badgen.net/static/Homepage/Wikipedia/blue)](https://snap.stanford.edu/jodie/) |
|  Mooc | [![Google drive](https://badgen.net/badge/Homepage/Mooc/cyan)](https://snap.stanford.edu/jodie/) |
| Bitcoin-Alpha  | [![Google drive](https://badgen.net/badge/Homepage/Bitcoin-Alpha/orange?)](https://snap.stanford.edu/data/soc-sign-bitcoin-alpha.html) |
| Bitcoin-OTC  | [![Google drive](https://badgen.net/badge/Homepage/Bitcoin-OTC/orange)](https://snap.stanford.edu/data/soc-sign-bitcoin-otc.html)|
| Amazon | [![Google drive](https://badgen.net/badge/Homepage/Amazon/yellow)](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)|






A dataset is a directory with the following structure:
  ```bash
  $ tree data
  NFTs_Dataset
  ├── mapping
  │   ├── nft_id_mapping
  │   └── wallet_id_mapping
  │
  ├── collections.csv
  └── opensea_nft_transactions.parquet
  
  
  ```


## Baselines

| **Baseline** |  **Paper** |  **Code** |
|----------|:--------:|:--------------------------------------------------------------:|
| **Radar** | <a href="https://www.ijcai.org/proceedings/2017/0299.pdf"><img src="https://cdn-icons-png.flaticon.com/512/337/337946.png" width="20"></a> | <a href="https://docs.pygod.org/en/latest/generated/pygod.detector.Radar.html"><img src="https://raw.githubusercontent.com/simple-icons/simple-icons/develop/icons/github.svg" width="20" style="filter: invert(1);"></a> |
| **DOMINANT** | <a href="https://epubs.siam.org/doi/10.1137/1.9781611975673.67"><img src="https://cdn-icons-png.flaticon.com/512/337/337946.png" width="20"></a> | <a href="https://docs.pygod.org/en/latest/generated/pygod.detector.DOMINANT.html"><img src="https://raw.githubusercontent.com/simple-icons/simple-icons/develop/icons/github.svg" width="20" style="filter: invert(1);"></a> |
| **GDN** | <a href="https://dl.acm.org/doi/10.1145/3442381.3449922"><img src="https://cdn-icons-png.flaticon.com/512/337/337946.png" width="20"></a> | <a href="https://github.com/kaize0409/Meta-GDN_AnomalyDetection/tree/main"><img src="https://raw.githubusercontent.com/simple-icons/simple-icons/develop/icons/github.svg" width="20" style="filter: invert(1);"></a> |
| **SemiGNN** | <a href="https://ieeexplore.ieee.org/document/8970829"><img src="https://cdn-icons-png.flaticon.com/512/337/337946.png" width="20"></a> | <a href="https://github.com/safe-graph/DGFraud/tree/master"><img src="https://raw.githubusercontent.com/simple-icons/simple-icons/develop/icons/github.svg" width="20" style="filter: invert(1);"></a> |
| **F-FADE** | <a href="https://dl.acm.org/doi/10.1145/3437963.3441806"><img src="https://cdn-icons-png.flaticon.com/512/337/337946.png" width="20"></a> | <a href="https://github.com/snap-stanford/F-FADE"><img src="https://raw.githubusercontent.com/simple-icons/simple-icons/develop/icons/github.svg" width="20" style="filter: invert(1);"></a> |
| **NetWalk** | <a href="https://dl.acm.org/doi/10.1145/3219819.3220024"><img src="https://cdn-icons-png.flaticon.com/512/337/337946.png" width="20"></a> | <a href="https://github.com/chengw07/NetWalk"><img src="https://raw.githubusercontent.com/simple-icons/simple-icons/develop/icons/github.svg" width="20" style="filter: invert(1);"></a> |
| **AddGraph** | <a href="https://www.ijcai.org/proceedings/2019/614"><img src="https://cdn-icons-png.flaticon.com/512/337/337946.png" width="20"></a> | <a href="https://github.com/Ljiajie/Addgraph"><img src="https://raw.githubusercontent.com/simple-icons/simple-icons/develop/icons/github.svg" width="20" style="filter: invert(1);"></a> |
| **DyRep** | <a href="https://openreview.net/pdf?id=HyePrhR5KX"><img src="https://cdn-icons-png.flaticon.com/512/337/337946.png" width="20"></a> | <a href="https://github.com/twitter-research/tgn/tree/master"><img src="https://raw.githubusercontent.com/simple-icons/simple-icons/develop/icons/github.svg" width="20" style="filter: invert(1);"></a> |
| **TADDY** | <a href="https://ieeexplore.ieee.org/document/9599560/"><img src="https://cdn-icons-png.flaticon.com/512/337/337946.png" width="20"></a> | <a href="https://github.com/yuetan031/TADDY_pytorch/tree/main"><img src="https://raw.githubusercontent.com/simple-icons/simple-icons/develop/icons/github.svg" width="20" style="filter: invert(1);"></a> |
| **JODIE** | <a href="https://dl.acm.org/doi/10.1145/3292500.3330895"><img src="https://cdn-icons-png.flaticon.com/512/337/337946.png" width="20"></a> | <a href="https://github.com/twitter-research/tgn/tree/master"><img src="https://raw.githubusercontent.com/simple-icons/simple-icons/develop/icons/github.svg" width="20" style="filter: invert(1);"></a> |
| **TGAT** | <a href="https://openreview.net/forum?id=rJeW1yHYwH"><img src="https://cdn-icons-png.flaticon.com/512/337/337946.png" width="20"></a> | <a href="https://github.com/StatsDLMathsRecomSys/Inductive-representation-learning-on-temporal-graphs"><img src="https://raw.githubusercontent.com/simple-icons/simple-icons/develop/icons/github.svg" width="20" style="filter: invert(1);"></a> |
| **TGN** | <a href="https://arxiv.org/abs/2006.10637"><img src="https://cdn-icons-png.flaticon.com/512/337/337946.png" width="20"></a> | <a href="https://github.com/twitter-research/tgn/tree/master"><img src="https://raw.githubusercontent.com/simple-icons/simple-icons/develop/icons/github.svg" width="20" style="filter: invert(1);"></a> |
| **SAD** | <a href="https://www.ijcai.org/proceedings/2023/256"><img src="https://cdn-icons-png.flaticon.com/512/337/337946.png" width="20"></a> | <a href="https://github.com/D10Andy/SAD"><img src="https://raw.githubusercontent.com/simple-icons/simple-icons/develop/icons/github.svg" width="20" style="filter: invert(1);"></a> |
| **SLADE** | <a href="https://dl.acm.org/doi/pdf/10.1145/3637528.3671845"><img src="https://cdn-icons-png.flaticon.com/512/337/337946.png" width="20"></a> | <a href="https://github.com/jhsk777/SLADE"><img src="https://raw.githubusercontent.com/simple-icons/simple-icons/develop/icons/github.svg" width="20" style="filter: invert(1);"></a> |
| **MHisCL** | <a href="https://doi.org/10.1016/j.knosys.2025.113049"><img src="https://cdn-icons-png.flaticon.com/512/337/337946.png" width="20"></a> | <a href="https://github.com/Yun-Fu/MHisCL"><img src="static/images/github-white_.svg" width="21" ></a> |


## Training
To train `Drone-Guard` on a dataset, run:
```bash
 python  train.py \
      --cfg <config-file>  \
      --exp <experiment-name>  \
      --gpu <experiment-name> 
```  
 For example, to train `Drone-Guard` on Ped2:

```bash
python train.py \
    --cfg config/wikipedia.yaml # To Train model with both normal and pseudo anomalies data
```

## Configuration
 * We use [YAML](https://yaml.org/) for configuration.
 * We provide a couple preset configurations.
 * Please refer to `config.py` for documentation on what each configuration does.

## Results 

<p align="center">
  <img src="static/images/wikipedia_auc_curve_std.png" width="30%">
  <img src="static/images/Mooc_AUROC_curve.png" width="30%">
  <img src="static/images/Bitcoin_Alpha_curve.png" width="30%">
</p>

<p align="center">
  <img src="static/images/Bitcoin_otc_curve.png" width="30%">
  <img src="static/images/Amazon_AUROC_curve.png" width="30%">
</p>
<p align="center">
Mean ROC curves of <b>T-ADTG</b> on the five benchmark datasets, averaged over 10 runs. Each panel reports the mean AUROC and its standard deviation.
</p>

## Citing
If you find our work useful, please consider citing:
```BibTeX
The paper has been submitted and is currently under review.

```

## Contact
For any question, please file an [issue](https://github.com/slitiWassim/NFT-Suspicious-Activity/issues) or contact:

    Wassim Sliti : wassim.sliti@upm.es

## Acknowledgement

This work was carried out within the STRAST Research Group at the Information Processing and Telecommunications Center (IPTC), Universidad Politécnica de Madrid, as part of the [ CEDAR ](https://cedar-heu-project.eu/)   project, funded by the Horizon Europe Programme (Grant Agreement No. 101135577). 
