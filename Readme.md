
# Adaptive NeRF Framework for City-Scale Emergency Awareness

This repository provides the reference implementation for the paper:

**An Adaptive NeRF Framework for City-Scale Emergency Awareness**  
Panagiotis Sklavos, Georgios Anestis, Antonios Deligiannakis  
Technical University of Crete  

📄 *Accepted at EIDWT 2026 (to appear)*

---

## Abstract
We present a city-scale adaptive Neural Radiance Field (NeRF) framework designed for time-critical emergency-awareness scenarios. The proposed system enables rapid online updates of large-scale NeRF reconstructions as new aerial data become available, without requiring full retraining. The framework combines meta-continual learning, spatial modularization, and Instant-NGP primitives to achieve fast adaptation, scalability, and robustness to catastrophic forgetting.

---

## Repository Structure

```text
.
├── configs/                # Experiment and training configurations
├── data/                   # Data storage and related modules
├── models/                 # Instant-NGP experts and container NeRF
├── training/               # Offline meta-learning and online adaptation
├── evals/                  # Online adaptation and evaluation components
├── viewer/                 # Viewer and real-time visualization tools
├── scripts/                # Training, evaluation, and utility scripts
├── requirements.txt
└── README.md
```

---

## Environment Setup (Installation)
To get started, clone the repository, create a Conda environment, and install the required dependencies.
Python 3.11 and CUDA 11.8 are verified.

### 1. Clone the repository
```bash
git clone https://github.com/psklavos1/adaptive-city-nerf.git
cd adaptive-city-nerf
```

### 2) Create the environmet.
We provide an example setup using conda.
Install the correct version of PyTorch and dependencies:
```bash
conda create -n nerfenv python=3.11 -y
conda activate nerfenv
conda install -y pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

### 3) Install tiny-cuda-nn.
Efficient Instant-NGP primitives require `tiny-cuda-nn.

Make sure CUDA is compatible:
```bash
conda install -y -c nvidia cuda-nvcc=11.8 cuda-cudart-dev=11.8
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
nvcc --version   # should say 11.8
```

Install using the official [NVLabs](https://github.com/NVlabs/tiny-cuda-nn?utm_source) instructions:
```bash
pip install --no-build-isolation -v "git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"
```
In case of failure assert all the provided [requirements](https://github.com/NVlabs/tiny-cuda-nn?utm_source) are satisfied 


---
To add resources for downloading data and model checkpoints!


---

## Dataset Preparation
<!-- todo: add url for DRZ and above for NVLabs -->
Our data preparation and handling draws inpiration from [MegaNeRF's](https://github.com/cmusatyalab/mega-nerf/tree/main) approach.

Our framework is oriented towards real-life applications, so we support any image captured dataset. We recommend using COLMAP for pose extraction. It is important to note that our pipeline expects geo-referenced/aligned data. 
COLMAP also provides the fallback of Manhattan world alignemnt as a fallback through the model aligner utility in the case that gps coordinates are not available. 
The alignment type can be selected to be enu or ecef in COLMAP. Both are acceptable via COLMAP but we recommend selecting ecef and allow internal ecef-to-enu transformation with our scripts for data preparation.

The following **scripts** are provided for dataset preparation and clustering:
- `/scripts/prepare_dataset.py` takes a COLMAP reconstruction and prepares it for use in our framework. It processes the raw COLMAP data into a format that is compatible with the training pipeline:

**Input Data.**
The input data must be structured as follows:
```text
data_path/
  ├── model/    # COLMAP sparse model (cameras.bin, images.bin, points3D.bin)
  └── images/   # All registered images used by the COLMAP model
```
Example Usage:
```bash
./scripts/prepare_dataset.py --data_path data/drz --output_path data/drz/out/prepared --val_split 0.3 --scale_strategy camera_max --ecef_to_enu --enu_ref median
```

> **Important**: This step requires **COLMAP** to be installed. Refer to the COLMAP [installation instructions](https://colmap.github.io/) for setup and [faqs](https://colmap.github.io/faq.html) for common additional info (e.g. model aligner)

- `scripts/create_clusters.py` generates spatial training partitions for NeRF experts using Voronoi-distance routing. Rays are assigned to one or more spatial clusters, producing per-cluster masks that define which data each expert is trained on. 

### Example Usage

```bash
 ./scripts/create_clusters.py --data_path data/drz/out/prepared --grid_dim 2 2 --cluster_2d --boundary_margin 1.05 --ray_samples 256 --center_pixels --scene_scale 1.1 --output g22_grid_bm105_ss11 --resume
```
---

### Configuration
Experiments can be configured either via **command-line arguments** or by
providing a **JSON configuration file**.

- Command-line arguments are defined in `common/args.py`
- Example JSON configurations are provided in `configs/`

The configuration controls all major aspects of the system, including:

- **Dataset settings**  
  Dataset paths, splits, scaling, ray sampling, and clustering masks.

- **Model architecture**  
  NeRF submodule structure, MLP depth and width, feature encoding parameters.

- **Optimization**  
  Optimizer type, learning rates, schedulers, and mixed-precision settings.

- **Training procedure**  
  Batch sizes, meta-learning inner/outer loop parameters, and checkpointing.

- **Hardware utilization**  
  GPU usage, chunk sizes, and memory-related parameters.

Users are encouraged to start from an existing configuration in `configs/`
and modify only the parameters of interest.


### 1) Offline Meta-Learning
In the offline phase, we perform meta-learning on the dataset, training the region-specific NeRF experts to learn a good parameter initialization.

Run the **offline meta-learning**:
```bash
python nerf_runner.py --op train --configPath configs/train.json
```

### 2) Online Runtime Adaptation
In the online phase, we perfrom runtime-adaptation on incoming views. This script assess model performance on metrics like **PSNR**, **SSIM**, and **LPIPS** after adapting for a certain amount of Test Time Optimization (TTO) steps. Additionally it renders the incoming views to compare visually compare with ground truth images. All results are stored at `logs/experiment_name`.

Run **runtime adaptation**:
```bash
python nerf_runner.py --op eval --checkpoint_path logs/example --prefix step20000 --use_stored_args --tto 128
```

> **Important**: This step also requires the usage of COLMAP to incorporate novel views.
[Register/localize new images into an existing reconstruction](https://colmap.github.io/faq.html) describes the steps to be done to update the existing reconcstruction. For our pipeline the forth step of bundle adjustment is not needed.

### 3) Viewer
The framework includes an interactive viewer built using `nerfview` and `viser`. The users can toggle between navigating a static scene and performing live visualization during runtime adaptation. Also can perform submodule isolation to monitor modularization.

Run the **viewer** with:
```bash
python nerf_runner.py --op view --checkpoint_path logs/example --prefix step20000
```
> **Note:** Viewer evaluation may reduce adaptation throughput due to shared GPU usage.

TODO: 1) Add Experiments reproduction via the configs. 2) Upload models and data for one click download. 3)
---

## Acknowledgments

This research is supported by the European Union’s Horizon 2020 research and innovation programme under grant agreement No. 101092749, project **CREXDATA**. We sincerely thank the members of the Deutsches Rettungsrobotik Zentrum (DRZ), for supporting the data acquisition and providing the aerial dataset used in this work.

> The citation will be added with final bibliographic details after the conference.