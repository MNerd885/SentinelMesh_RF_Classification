# SentinelMesh_RF_Classification 

Code and technical annex for the report "Sentinel Mesh: Batteryless,
Camera-Free Presence Sensing with Intermittent Computing" (HiPEAC Student
Challenge 2027, Option A). This is the repository cited as reference [27] in
the report. Everything here is a design-stage study on simulated data; no
hardware has been built.

## What is in this repository

| Folder | Content | Report sections |
|---|---|---|
| `dataset-a121/` | Synthetic Acconeer A121 dataset: generator, CSV files and the pipeline write-up | Sec. IV |
| `classification-src/` | Random Forest training and evaluation scripts | Sec. IV, VII-D |
| `model/` | Classifier exported as a C header (emlearn) | Sec. IV-C |
| `Animations/` | Animations of the simulated motions and their range-Doppler maps | Sec. IV-B |
| `systems-src/` | Checkpoint fault-injection model, energy simulation and the first firmware draft | Sec. V to VII, Annex A |
| `docs/` | Technical annex of the report (Annex A.1 to A.12) | Annex A |

## Sensing and classification

### How to run it on every PC

First thing first a virtual Python environment must be created and activated.

### Linux

```bash
cd SentinelMesh_RF_CLassification

python -m venv <VENV-NAME>

source <VENV-NAME>/bin/activate

```

Once that's done you need to install dependencies for the project

```bash
    pip install -r requirements.txt
```

and you're good to go!!
The explanation of what I did is this one [Pipeline for the project and functioning of the Acconeer A121](dataset-a121/data_gathering.md)

Run the scripts from the repository root, because they read the dataset with
paths relative to it:

```bash
python classification-src/classification_performance.py
```

## Intermittent runtime and energy model

The two validation models need no extra packages (matplotlib only for the
figure). Details, results and known limits are in
[systems-src/README.md](systems-src/README.md).

```bash
cd systems-src
gcc -O2 -Wall -o ckpt_sim ckpt_sim.c && ./ckpt_sim
python3 energy_sim.py fig
```

## Technical annex

[docs/Sentinel_Mesh_Technical_Annex.pdf](docs/Sentinel_Mesh_Technical_Annex.pdf)
holds Annex A of the report: the assumption register, the checkpoint format,
the failure cases, the LoRa airtime table, the energy balance, the pin map and
the verification set-up.


EOF
