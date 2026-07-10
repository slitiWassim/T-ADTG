# A Time-Aware Self-Supervised Framework for Anomaly Detection in Temporal Graphs
This is the code for **[A Time-Aware Self-Supervised Framework for Anomaly Detection in Temporal Graphs](https://github.com/slitiWassim/NFT-Suspicious-Activity)** .

### [Project](https://slitiwassim.github.io/NFT-Suspicious-Activity/) | [Dataset](https://drive.upm.es/s/sLgeSrNxMEzXaEB?openfile=true) | [Paper]()
 


<p align="center">
  <a href="static/images/paper_.png" target="_blank">
    <img
      src="static/images/paper_.png"
      width="70%"
      style="border: 2px solid rgb(201, 196, 196);"
    >
  </a>
</p>

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

## Setup
The code can be run under any environment with Python 3.9.25 and above.
(It may run with lower versions, but we have not tested it).


Clone this repo:

    git clone https://github.com/slitiWassim/T-ADTG.git
    cd T-ADTG/


Install the required packages:

    pip install -r requirements.txt
  


We evaluate `T-ADTG` on:
| Dataset | Link                                                                                  |
|--|---------------------------------------------------------------------------------------|
| UCSD Ped2 | [![Google drive](https://badgen.net/static/Homepage/Ped2/blue)](http://www.svcl.ucsd.edu/projects/anomaly/dataset.html) |
| CUHK Avenue | [![Google drive](https://badgen.net/badge/Homepage/Avenue/cyan)](http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html) |
| ShanghaiTech | [![Google drive](https://badgen.net/badge/Homepage/ShanghaiTech/green?)](https://svip-lab.github.io/dataset/campus_dataset.html) |
| Drone-Anomaly | [![Google drive](https://badgen.net/badge/Homepage/Drone-Anomaly/yellow)](https://www.kaggle.com/datasets/dayaalex/drone-anomaly)|
| Drone-Anomaly | [![Google drive](https://badgen.net/badge/Homepage/Drone-Anomaly/yellow)](https://www.kaggle.com/datasets/dayaalex/drone-anomaly)|
| Drone-Anomaly | [![Google drive](https://badgen.net/badge/Homepage/Drone-Anomaly/yellow)](https://www.kaggle.com/datasets/dayaalex/drone-anomaly)|




<p align="center">
<b>Figure.</b> Mean ROC curves of <b>T-ADTG</b> on the five benchmark datasets, averaged over 10 runs. Each panel reports the mean AUROC and its standard deviation, with the diagonal indicating random performance.
</p>


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



### Data description
| Descriptions | Statistics                                                                                  |
|--|---------------------------------------------------------------------------------------|
|Start date(dd-mm-yyyy,UTC) | 23-06-2017 21:05 |
|End date (dd-mm-yyyy, UTC)| 22-12-2023 19:06 |
|Number of NFT collections | 1,746,379 |
|Number of NFT tokens |41,292,572 |
|Number of account addresses| 7,062,831 |
|Number of transactions |76,300,244  |
|Chains | 10 |



## Baselines

| Baseline | Paper | Code |
|----------|-------|------|
| Radar | [Paper](https://paper-link) | [Code](https://code-link) |
| DOMINANT | [Paper](https://paper-link) | [Code](https://code-link) |
| GDN | [Paper](https://paper-link) | [Code](https://code-link) |
| SemiGNN | [Paper](https://paper-link) | [Code](https://code-link) |
| F-FADE | [Paper](https://paper-link) | [Code](https://code-link) |
| NetWalk | [Paper](https://paper-link) | [Code](https://code-link) |
| AddGraph | [Paper](https://paper-link) | [Code](https://code-link) |
| DyRep | [Paper](https://paper-link) | [Code](https://code-link) |
| TADDY | [Paper](https://paper-link) | [Code](https://code-link) |
| JODIE | [Paper](https://paper-link) | [Code](https://code-link) |
| TGAT | [Paper](https://paper-link) | [Code](https://code-link) |
| TGN | [Paper](https://paper-link) | [Code](https://code-link) |
| SAD | [Paper](https://paper-link) | [Code](https://code-link) |
| SLADE | [Paper](https://paper-link) | [Code](https://code-link) |
| MHisCL | [Paper](https://paper-link) | [Code](https://code-link) |


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
