# EaDRL Training

## Create Environment

```bash
conda env create -f environment.yaml
conda activate gt_drl
```

## Train

```bash
cd EaDRL

python train.py \
  --cuda 0 \
  --algorithm gt_drl \
  --job_centric \
  --new_job
```

```bash
cd EaDRL/dynamicAMR/mapf

python run_training.py
```

## Third-party components

The dynamicAMR/mapf module contains adapted third-party MAPF components with their original license retained.

The third-party license information does not indicate the identity or affiliation of the authors of this submission.