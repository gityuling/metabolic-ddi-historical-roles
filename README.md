# Historical Drug Roles and Molecular Structure for Predicting Metabolic Drug–Drug Interactions

## Verify the paper's main comparison

With Python 3.10 or later, install the evaluation dependencies and point the verifier at the companion benchmark ZIP. It checks the mean and sample SD of all twelve four-class accuracies and the per-seed opposite-direction success rates from the original predictions. The optional plot uses percentages.

```sh
python -m pip install -r requirements-evaluation.txt
python verify_release.py metabolic-ddi-benchmark-v1.0.zip --plot accuracy.pdf
```

The ZIP contains the fit, validation and test pair manifests, the five final probability matrices for each method on each dataset, and numerical results for the main and supplementary analyses. See its README for class definitions and identifier derivation.

The fit, validation and test cohorts were rebuilt from the upstream files and compared with the experimental manifests, including their row order. The archived predictions were validated on CPU; full GPU model fitting was not repeated for this release.

## Original releases and local input layout

Run the commands below from the project root with `MC=manuscript/pharmaceutics/analysis/main_comparison` and `NM=MFP-DDI/novel_model` set in your shell.

- Dataset 1: obtain the [MeTDDI data archive](https://github.com/LabWeng/MeTDDI#data-and-weights). Copy the three CSV files from `data/Classification/UnseenDDIs` to `MFP-DDI/data/`. Run `python "$MC/prepare_metddi.py"`.
- Dataset 2: download [event_medium.db](https://doi.org/10.5281/zenodo.18907274) to `$MC/event_medium_native/`. Run `python "$MC/event_medium_native/prepare_split.py"`.
- Dataset 3: put the [MecDDI](https://mecddi.idrblab.net/download) drug information and metabolic-interactions CSVs in `$MC/mecddi_native/raw/` as `drugs.csv` and `metabolism.csv`. From the [PKAG-DDI release](https://huggingface.co/datasets/acai233/pkag-ddi/tree/f9aebd8173a9b8a9996f0d4390946d904b8d103b/data/MecDDI), copy `all_ddi_addSMIELS_raw.pkl`, `all_ddi_addSMIELS.pkl` (rename to `compact.pkl`) and the three seed-0 random split text files into the same directory. Run `python "$MC/mecddi_native/prepare_split.py"`.

## Training

Training additionally uses RDKit, scikit-learn, SciPy, PyTorch with CUDA, PyTorch Geometric and CatBoost (Dataset 1 used 1.2.10), plus the dependencies of the upstream baseline being fitted. Fetch the authors' [MFPs source](https://github.com/manelgil3/DDI_Addressing_Overcomplexity) at commit `997cc5fc03ce0f60f8fcbb93347f56f072b16c72` in `MFP-DDI/upstream/` and the [IAMV-DDI source and pretrained artifacts](https://github.com/mikael-0812/IAMV-DDI) in `IAMV-DDI/`. IAMV-DDI inputs under `$MC/inputs/` are its two encoder checkpoints (`2d.pth`, `3d.pt`), two structure caches (`2d_cache.pt`, `3d_cache.pt`) and `id_map.csv`.

1. Dataset 1: run each of `neighbors.py`, `latent.py` and `kernel.py` under `$NM` twice, with the arguments `prepare` then `test`. Run `python "$MC/metddi_matched/run.py"` for MFPs and the full model. Run `python "$MC/tokens.py"` then `python "$MC/iamv_extended/run.py"` for IAMV-DDI.
2. Datasets 2 and 3: for each target folder (`$MC/event_medium_native` or `$MC/mecddi_native`), run `python "$MC/event_medium_native/run_event.py" METHOD TARGET_FOLDER` with METHOD taking `tokens`, `mfps`, `graph` and `iamv` in turn. Run the target folder's `score_event.py` or `score_mecddi.py` to apply the structural fallback and score the predictions.
3. For the MeTDDI baseline use its [released code](https://github.com/LabWeng/MeTDDI) in `$NM/final_suite/MeTDDI/` and its original `preprocessed_drug_info.npy` in `$NM/final_suite/met_assets/`. Use a separate TensorFlow environment (study versions: TensorFlow 2.22.0rc1 and tf-keras 2.22.0rc0, legacy Keras enabled). Run `python "$MC/metddi_native/train_matched.py" SEED` for Dataset 1. For the other datasets run `python "$MC/metddi_native/prepare.py"`, then `python "$MC/metddi_native/train.py" DATASET --seed SEED`, where DATASET is the target folder name. Use seeds 0–4.

The fit scripts are supplied for inspection and reruns with the original upstream inputs. The CPU verification uses the frozen manuscript predictions and does not require a GPU or pretrained weights.
