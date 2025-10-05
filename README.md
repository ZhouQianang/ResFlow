# ResFlow
Event-based High Temporal Resolution Flow
This repository contains the implementation for the paper: ResFlow: Fine-tuning Residual Optical Flow for Event-based High Temporal Resolution Motion Estimation

## Dataset Preparation
Prepare the DSEC dataset in the 100ms-16bins.
Note: the first bin is not used, as 16bins correspond to 15 time intervals.

## Training
1. Traing the LTR part. The LTR stage is similar to TMA, but differs in the voxel bin setting (15 for TMA vs. 16 in our method)
2. Traing the HTR part. Starting from the pretrained LTR model, run the HTR training useing:
```
bash scripts/train.sh
```

## Testing
- Prepare the event-data of DSEC-Flow-test as required. our method does not use raw event data during trainging. 
- The Event-FWL metric implementation is provided in the `utils` directory, and and example od its usage is provided in `train_Dsec_Separate.py`

## Acknowledgment
This work builds upon several excellent open-source projects:
- TMA
- E-RAFT
- RAFT
