
# Adaptive NeRF Framework for City-Scale Emergency Awareness

This repository provides the reference implementation for the paper:

**An Adaptive NeRF Framework for City-Scale Emergency Awareness**  
Panagiotis Sklavos, Georgios Anestis, Antonios Deligiannakis  
Technical University of Crete  

📄 *Accepted at EIDWT 2026 (to appear)*

---

## Overview
![Overview of the Adaptive NeRF pipeline](pipeline.png)


We present a city-scale adaptive Neural Radiance Field (NeRF) framework designed for time-critical emergency-awareness scenarios. The proposed system enables rapid online updates of large-scale NeRF reconstructions as new aerial data become available, without requiring full retraining. The framework combines meta-continual learning, spatial modularization, and Instant-NGP primitives to achieve fast adaptation, scalability, and robustness to catastrophic forgetting.


```text
.
├── common/                     # Shared utilities (logging, helpers)
├── configs/                    # Demo configuration files
├── data/                       # Dataset storage and parsing
├── logs/                       # Execution Outputs, logs, and checkpoints 
├── models/                     # NeRF models and architectures
├── nerfs/                      # NeRF-specific helpers
├── pipelines/                  # Meta-training and runtime adaptation pipelines
├── scripts/                    # Internal execution scripts for data preparation or logging statistics
├── viewer/                     # Interactive NeRF viewer
├── nerf_runner.py              # Entry point for executing NeRF operations
├── utils.py                    # Entry point utilities
├── requirements.txt            # Python dependencies
└── README.md                   # Repo documentation
```

---

## Environment Setup
To get started, clone the repository, create a Conda environment, and install the required dependencies.
Python 3.11 and CUDA 11.8 are verified for compatibility.

#### 1. Clone the repository
```bash
git clone https://github.com/psklavos1/adaptive-city-nerf.git
cd adaptive-city-nerf
```

#### 2) Create the environmet.
We provide an example setup using conda.
Install the correct version of PyTorch and dependencies:
```bash
conda create -n nerfenv python=3.11 -y
conda activate nerfenv
conda install -y pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

#### 3) Install tiny-cuda-nn.
Efficient Instant-NGP primitives require `tiny-cuda-nn`.

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
In case of failure assert all the provided [requirements](https://github.com/NVlabs/tiny-cuda-nn?utm_source) are satisfied.

---

## Data Preparation
Our data preparation and ray routing draw inpiration from [MegaNeRF's](https://github.com/cmusatyalab/mega-nerf/tree/main) approach.

The framework is oriented towards real-life applications supporting any image-based dataset with known or recoverable camera poses that are geo-referenced. We recommend using [COLMAP](https://colmap.github.io/) for pose estimation. if GPS coordinates are unavailable, COLMAP’s model alignment utilities (e.g., Manhattan-world alignment) may be used as a fallback.

COLMAP exports data either in ECEF or ENU world alignment. We recommend exporting in ECEF from COMLAP's model aligner and performing ECEF→ENU conversion internally using the provided scripts for improved compatibility and control, yet direct ENU exports are also fine.

> **Important**: This step requires **COLMAP** to be installed. Refer to the COLMAP [installation guide](https://colmap.github.io/) for setup and [faqs](https://colmap.github.io/faq.html) for additional info (e.g. model aligner)

The following **scripts** are provided for dataset preparation and clustering:
- `/scripts/prepare_dataset.py` converts a COLMAP reconstruction into the framework’s expected format. It normalizes scene scale for stable training and optionally converts poses to a common ENU frame, storing all outputs using the framework’s internal coordinate conventions.

**Input Data.**
The input data must be structured as follows:
```text
data_path/
  ├── model/    # COLMAP sparse model (cameras.bin, images.bin, points3D.bin)
  └── images/   # All registered images used by the COLMAP model
```
### Example
```bash
./scripts/prepare_dataset.py --data_path data/drz --output_path data/drz/out/prepared --val_split 0.3 --scale_strategy camera_max --ecef_to_enu --enu_ref median
```

- `scripts/create_clusters.py` generates spatial training partitions for NeRF experts using Voronoi-distance routing. For each image, it assigns rays to one or more centroids (optionally with boundary overlap) and writes per-cluster mask files that define which rays/images each expert trains on. 2D Clustering is strongly recommended as it reduces computation with no significant side-effects.

### Example

```bash
 ./scripts/create_clusters.py --data_path data/drz/out/prepared --grid_dim 2 2 --cluster_2d --boundary_margin 1.05 --ray_samples 256 --center_pixels --scene_scale 1.1 --output g22_grid_bm105_ss11 --resume
```
---

## Configuration
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

---
## Usage
The repo supports meta-training a model on an offline dataset, evaluating a meta-trained model to streamed incoming views and finally a viewer for rendering the model and or monitoring live adaptation.

### 1) Offline Meta-Learning
In the offline phase, we perform meta-learning on the dataset, training the region-specific NeRF experts to learn a good parameter initialization.

Run the **offline meta-learning**:
```bash
python nerf_runner.py --op train --configPath configs/train.json
```

### 2) Runtime Evaluation

In the online phase, we perform runtime adaptation on newly acquired views. To evaluate adaptation quality we compute metrics, such as **PSNR**, **SSIM**, and **LPIPS** after a specified number of **Test-Time Optimization (TTO)** steps. Incoming views are also rendered for visual comparison with ground truth images. All results are stored under `logs/<experiment_name>/`.

#### Required workflow (must be followed in this order)

1. **Update the COLMAP reconstruction with new images**  
   New images must first be registered/localized into the existing COLMAP model.  
   > See: [Register/localize new images into an existing reconstruction](https://colmap.github.io/faq.html)  
   For this pipeline, the final **bundle adjustment** step is **not required**.

2. **Update the prepared dataset with only the newly registered images**  
   After updating COLMAP, extend the already-prepared dataset without recomputing origin/scale or modifying existing train/val splits:

   ```bash
   ./scripts/update_dataset.py --update_model_path <colmap_model_dir> --image_path <images_dir> --prepared_dir <prepared_dir> --batch_tag <batch_name>
   ```
   New data are written under `<prepared_dir>/continual/<batch_tag>/`
3.  Run **Evaluation**
    ```bash
    python nerf_runner.py --op eval --checkpoint_path logs/example --prefix step20000 --use_stored_args --tto 128
    ```
   > **Important**: The above execution is intended for evaluating model performance
    using reconstruction metrics. For practical runtime adaptation monitoring that
    reflects realistic application scenarios, the viewer should be used instead, as
    it enables direct NeRF rendering and interactive inspection.

### 3) Viewer
The framework includes an interactive viewer built using [`nerfview`](https://github.com/nerfstudio-project/nerfview). The users can toggle between navigating a static scene and performing live visualization during runtime-adaptation. Also can perform submodule isolation to monitor modularization.

Run the **viewer** with:
```bash
python nerf_runner.py --op view --checkpoint_path logs/example --prefix step20000
```

To monitor NeRF rendering the web-based viewer a link is provided at the console for users to follow. 

> **Note:** Live adaptation on viewer may reduce throughput due to shared GPU usage. 

---

## Experiment Data & Model
The demo data for the experiments are organized in `data/drz/out/example` for rapid testing. Those data are produced performing all  data preparation steps, along with the additional update for performing runtime adaptation.

The model checkpoint with 4 experts used in the paper: [`checkpoint`](https://github.com/psklavos1/adaptive-city-nerf/releases/tag/v1.0/4_experts.zip)

After downloading, extract into: `logs/example`

---

## Citation & License

If you use this framework in your research, please cite our paper:

```bibtex
@inproceedings{sklavos2026adaptivenerf,
  title     = {An Adaptive NeRF Framework for City-Scale Emergency Awareness},
  author    = {Panagiotis Sklavos and Georgios Anestis and Antonios Deligiannakis},
  booktitle = {Proceedings of the 14th International Conference on Emerging Internet, Data \& Web Technologies (EIDWT 2026)},
  year      = {2026},
  note      = {Accepted for publication}
}
```
> **Note**: The citation will be updated upon publication.

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---
## Acknowledgments
This research is supported by the European Union’s Horizon 2020 research and innovation programme under grant agreement No. 101092749, project [**CREXDATA**](https://crexdata.eu/). We sincerely thank the members of the [Deutsches Rettungsrobotik Zentrum (DRZ)](https://rettungsrobotik.de/), for supporting the data acquisition and providing the aerial dataset used in this work.

