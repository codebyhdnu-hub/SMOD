<div align="center">

<h1>Leveraging Dark Knowledge for Intrinsic Multimodal Out-of-Distribution Detection</h1>

---

</div>


### Prepare Datasets
Our code is evaluated on MultiOOD benchmark, which is constructed from five public action recognition datasets:
HMDB51, UCF101, EPIC-Kitchens, HAC, and Kinetics-600.
To reproduce our results, please download and prepare the datasets following the MultiOOD benchmark instructions.
You can find the dataset preparation details and download links in the official MultiOOD repository [link](https://github.com/donghao51/MultiOOD).


## Methodology
<div style="text-align:left">
<img src="imgs/smod.drawio.png"  width="80%" height="100%">
</div>

---

An overview of the proposed framework for Multimodal OOD Detection.

## Code
The code was tested using `Python 3.10.4`, `torch 1.11.0+cu113` and `NVIDIA-A100 GPU`. More dependencies are in `requirement.txt`.

### Prepare

#### Download Pretrained Weights
1. Download SlowFast model for RGB modality [link](https://download.openmmlab.com/mmaction/recognition/slowfast/slowfast_r101_8x8x1_256e_kinetics400_rgb/slowfast_r101_8x8x1_256e_kinetics400_rgb_20210218-0dd54025.pth) and place under the `HMDB-rgb-flow/pretrained_models` and `EPIC-rgb-flow/pretrained_models` directory
   
2. Download SlowOnly model for Flow modality [link](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_8x8x1_256e_kinetics400_flow/slowonly_r50_8x8x1_256e_kinetics400_flow_20200704-6b384243.pth) and place under the `HMDB-rgb-flow/pretrained_models` and `EPIC-rgb-flow/pretrained_models` directory

3. Download Audio model [link](http://www.robots.ox.ac.uk/~vgg/data/vggsound/models/H.pth.tar), rename it as `vggsound_avgpool.pth.tar` and place under the `HMDB-rgb-flow/pretrained_models` and `EPIC-rgb-flow/pretrained_models`  directory
   
### Multimodal Near-OOD Benchmark
#### HMDB51 25/26

<details>
<summary>Click for details...</summary>

```
cd HMDB-rgb-flow/
```

Train the Near-OOD model for HMDB:

```
python SMOD_train_video_flow.py --near_ood --dataset 'HMDB' --lr 0.0001 --seed 0 --bsz 16 --num_workers 8 --nepochs 10 --appen 'CEall_FT_' --save_best --save_checkpoint --resumef '../HMDB-rgb-flow/models/HMDB_near_ood_baseCEall.pt'  --datapath '/path/to/HMDB51/' 
```

Save the evaluation files for HMDB (to save evaluation files for ASH or ReAct, you should also run following line with options `--use_ash` or `--use_react`):
```
python test_video_flow.py --bsz 16 --num_workers 2 --near_ood --dataset 'HMDB' --appen 'CEall_FT_' --resumef '/path/to/HMDB_near_ood_CEall_FT_.pt'
```

Evaluation for HMDB (change `--postprocessor` to different score functions, for VIM you should also pass `--resume_file checkpoint.pt`, where checkpoint.pt is the trained checkpoint):
```
python eval_video_flow_near_ood.py --postprocessor msp --appen 'CEall_FT_' --dataset 'HMDB' --path 'HMDB-rgb-flow/'
```

</details>


#### UCF101 50/51
<details>
<summary>Click for details...</summary>

```
cd HMDB-rgb-flow/
```

Train the Near-OOD model for UCF:

```
python SMOD_train_video_flow.py --near_ood --dataset 'UCF' --lr 0.0001 --seed 0 --bsz 16 --num_workers 8 --nepochs 10 --appen 'CEall_FT_' --save_best --save_checkpoint --save_checkpoint --resumef '../HMDB-rgb-flow/models/UCF_near_ood_baseCEall.pt' --datapath '/path/to/UCF101/' 
```


Save the evaluation files for UCF (to save evaluation files for ASH or ReAct, you should also run following line with options `--use_ash` or `--use_react`):
```
python test_video_flow.py --bsz 16 --num_workers 2 --near_ood --dataset 'UCF' --appen 'CEall_FT_' --resumef '/path/to/UCF_near_ood_CEall_FT_.pt'
```

Evaluation for UCF (change `--postprocessor` to different score functions, for VIM you should also pass `--resume_file checkpoint.pt`, where checkpoint.pt is the trained checkpoint):
```
python eval_video_flow_near_ood.py --postprocessor msp --appen 'CEall_FT_' --dataset 'UCF' --path 'HMDB-rgb-flow/'
```

</details>

#### EPIC-Kitchens 4/4
<details>
<summary>Click for details...</summary>

```
cd EPIC-rgb-flow/
```

Train the Near-OOD model for EPIC:

```
python SMOD_train_video_flow_epic.py --dataset 'EPIC' --lr 0.0001 --seed 0 --bsz 16 --num_workers 8 --nepochs 10 --appen 'CEall_FT_' --save_best --save_checkpoint --save_checkpoint --resumef '../HMDB-rgb-flow/models/EPIC_near_ood_baseCEall.pt' --datapath '/path/to/EPIC-Kitchens/' 
```

Save the evaluation files for EPIC (to save evaluation files for ASH or ReAct, you should also run following line with options `--use_ash` or `--use_react`):
```
python test_video_flow_epic.py --bsz 16 --num_workers 2  --ood_dataset 'EPIC' --appen 'CEall_FT_' --resumef '/path/to/EPIC_near_ood_CEall_FT_.pt'
```

Evaluation for EPIC (change `--postprocessor` to different score functions, for VIM you should also pass `--resume_file checkpoint.pt`, where checkpoint.pt is the trained checkpoint):
```
python eval_video_flow_near_ood.py --postprocessor msp --appen 'CEall_FT_' --dataset 'EPIC' --path 'EPIC-rgb-flow/'
```

</details>

#### Kinetics-600 129/100
<details>
<summary>Click for details...</summary>

```
cd HMDB-rgb-flow/
```

Train the Near-OOD model for Kinetics:

```
python SMOD_train_video_flow.py --near_ood --dataset 'Kinetics' --lr 0.0001 --seed 0 --bsz 16 --num_workers 8 --nepochs 10 --appen 'CEall_FT_' --save_best --save_checkpoint --save_checkpoint --resumef '../Kinetics-rgb-flow/models/HMDB_near_ood_baseCEall.pt' --datapath '/path/to/Kinetics-600/' 
```

Save the evaluation files for Kinetics (to save evaluation files for ASH or ReAct, you should also run following line with options `--use_ash` or `--use_react`):
```
python test_video_flow.py --bsz 16 --num_workers 2 --near_ood --dataset 'Kinetics' --appen 'CEall_FT_' --resumef '/path/to/Kinetics_near_ood_CEall_FT_.pt'
```

Evaluation for Kinetics (change `--postprocessor` to different score functions, for VIM you should also pass `--resume_file checkpoint.pt`, where checkpoint.pt is the trained checkpoint):
```
python eval_video_flow_near_ood.py --postprocessor msp --appen 'CEall_FT_' --dataset 'Kinetics' --path 'HMDB-rgb-flow/'
```

</details>


### Multimodal Far-OOD Benchmark
#### HMDB51 as ID

<details>
<summary>Click for details...</summary>

```
cd HMDB-rgb-flow/
```

Train the Far-OOD model for HMDB:

```
python SMOD_train_video_flow.py --dataset 'HMDB' --lr 0.0001 --seed 0 --bsz 16 --num_workers 8 --nepochs 10 --appen 'baseCEall_far_' --save_best --save_checkpoint --resumef '../HMDB-rgb-flow/models/HMDB_far_ood_baseCEall.pt' -datapath '/path/to/HMDB51/' 
```

Save the evaluation files for HMDB (to save evaluation files for ASH or ReAct, you should also run following line with options `--use_ash` or `--use_react`, same for other datasets):
```
python test_video_flow.py --bsz 16 --num_workers 2 --dataset 'HMDB' --appen 'baseCEall_far_' --resumef '/path/to/HMDB_far_ood_baseCEall_far_.pt'
```

Save the evaluation files for UCF:
```
python test_video_flow.py --bsz 16 --num_workers 2 --far_ood --dataset 'HMDB' --ood_dataset 'UCF' --appen 'baseCEall_far_' --resumef '/path/to/HMDB_far_ood_baseCEall_far_.pt'
```

Save the evaluation files for HAC:
```
python test_video_flow.py --bsz 16 --num_workers 2 --far_ood --dataset 'HMDB' --ood_dataset 'HAC' --appen 'baseCEall_far_' --resumef '/path/to/HMDB_far_ood_baseCEall_far_.pt'
```

Save the evaluation files for Kinetics:
```
python test_video_flow.py --bsz 16 --num_workers 2 --far_ood --dataset 'HMDB' --ood_dataset 'Kinetics' --appen 'baseCEall_far_' --resumef '/path/to/HMDB_far_ood_baseCEall_far_.pt'
```

Save the evaluation files for EPIC:
```
cd EPIC-rgb-flow/
```
```
python test_video_flow_epic.py --bsz 16 --num_workers 2 --far_ood --dataset 'HMDB' --ood_dataset 'EPIC' --appen 'baseCEall_far_' --resumef '/path/to/HMDB_far_ood_baseCEall_far_.pt'
```


Evaluation for UCF (change `--postprocessor` to different score functions, for VIM you should also pass `--resume_file checkpoint.pt`, where checkpoint.pt is the trained checkpoint, change `--ood_dataset` to UCF, EPIC, HAC, or Kinetics):
```
python eval_video_flow_far_ood.py --postprocessor msp --appen 'baseCEall_far_' --dataset 'HMDB' --ood_dataset 'UCF' --path 'HMDB-rgb-flow/'
```

</details>


#### Kinetics as ID

<details>
<summary>Click for details...</summary>

```
cd HMDB-rgb-flow/
```

Train the Far-OOD model for Kinetics:

```
python train_video_flow.py --dataset 'Kinetics' --lr 0.0001 --seed 0 --bsz 16 --num_workers 8 --nepochs 10 --appen 'baseCEall_far_' --save_best --save_checkpoint --resumef '../HMDB-rgb-flow/models/Kinetics_far_ood_baseCEall.pt' --datapath '/path/to/Kinetics-600/' 
```

Save the evaluation files for Kinetics (to save evaluation files for ASH or ReAct, you should also run following line with options `--use_ash` or `--use_react`, same for other datasets):
```
python test_video_flow.py --bsz 16 --num_workers 2 --dataset 'Kinetics' --appen 'baseCEall_far_' --resumef '/path/to/Kinetics_far_ood_baseCEall_far_'
```

Save the evaluation files for HMDB:
```
python test_video_flow.py --bsz 16 --num_workers 2 --far_ood --dataset 'Kinetics' --ood_dataset 'HMDB' --appen 'baseCEall_far_' --resumef '/path/to/Kinetics_far_ood_baseCEall_far_.pt'
```

Save the evaluation files for UCF:
```
python test_video_flow.py --bsz 16 --num_workers 2 --far_ood --dataset 'Kinetics' --ood_dataset 'UCF' --appen 'baseCEall_far_' --resumef '/path/to/Kinetics_far_ood_baseCEall_far_.pt'
```

Save the evaluation files for HAC:
```
python test_video_flow.py --bsz 16 --num_workers 2 --far_ood --dataset 'Kinetics' --ood_dataset 'HAC' --appen 'baseCEall_far_' --resumef '/path/to/Kinetics_far_ood_baseCEall_far_.pt'
```

Save the evaluation files for EPIC:
```
cd EPIC-rgb-flow/
```
```
python test_video_flow_epic.py --bsz 16 --num_workers 2 --far_ood --dataset 'Kinetics' --ood_dataset 'EPIC' --appen 'baseCEall_far_' --resumef '/path/to/Kinetics_far_ood_baseCEall_far_.pt'
```


Evaluation for UCF (change `--postprocessor` to different score functions, for VIM you should also pass `--resume_file checkpoint.pt`, where checkpoint.pt is the trained checkpoint, change `--ood_dataset` to UCF, EPIC, HAC, or HMDB):
```
python eval_video_flow_far_ood.py --postprocessor msp --appen 'baseCEall_far_' --dataset 'Kinetics' --ood_dataset 'UCF' --path 'HMDB-rgb-flow/'
```

</details>

### Multimodal Near-OOD Benchmark with Video, Audio, and Optical Flow

#### Kinetics-600 129/100
<details>
<summary>Click for details...</summary>

```
cd HMDB-rgb-flow/
```

Train the Near-OOD model for Kinetics:

```
python SMOD_train_video_flow_audio.py --near_ood --dataset 'Kinetics' --lr 0.0001 --seed 0 --bsz 16 --num_workers 8 --nepochs 10 --appen 'baseCEall_Allmodal_' --save_best --save_checkpoint --resumef '../HMDB-rgb-flow/models/Kinetics_near_ood_vfa_baseCEall.pt' --datapath '/path/to/Kinetics-600/' 
```

Save the evaluation files for Kinetics (to save evaluation files for ASH or ReAct, you should also run following line with options `--use_ash` or `--use_react`):
```
python test_video_flow_audio.py --bsz 16 --num_workers 2 --near_ood --dataset 'Kinetics' --appen 'baseCEall_Allmodal_' --resumef '/path/to/Kinetics_near_ood_baseCEall_Allmodal_.pt'
```

Evaluation for Kinetics (change `--postprocessor` to different score functions, for VIM you should also pass `--resume_file checkpoint.pt`, where checkpoint.pt is the trained checkpoint):
```
python eval_video_flow_near_ood.py --postprocessor msp --appen 'baseCEall_Allmodal_' --dataset 'Kinetics' --path 'HMDB-rgb-flow/'
```

</details>


## Citation

If you find our work useful in your research please consider citing our paper:

```
@article{smod,
	author   = {},
	title    = {{Leveraging Dark Knowledge for Intrinsic Multimodal Out-of-Distribution Detection}},
    journal  = {arXiv preprint arXiv:},
	year     = {2025},
}
```

## Related Projects

[MultiOOD](https://github.com/donghao51/MultiOOD/): MultiOOD: Scaling Out-of-Distribution Detection for Multiple Modalities

[DPU](https://github.com/lili0415/DPU-OOD-Detection/): Dynamic Prototype Updating for Multimodal Out-of-Distribution Detection

[Feature Mixing](https://github.com/mona4399/FeatureMixing): Extremely Simple Multimodal Outlier Synthesis for Out-of-Distribution Detection and Segmentation


## Acknowledgement

Many thanks to the open-source projects [MultiOOD](https://github.com/donghao51/MultiOOD/) and [DPU](https://github.com/lili0415/DPU-OOD-Detection/).
