# SemSeg4RS-MRMS

## Pipeline Commands

This project implements a multi-sensor remote sensing semantic segmentation pipeline.

Current sensors:

- `spot`
- `planetscope`

Current sensor setups:

- `spot`
- `planetscope`
- `mixed`

Current models:

- `crossearth`
- `deeplabv3plus`
- `segformer_sae`
- `dofa`

Unified training labels:

```text
0   = sealed_soil
1   = non_sealed_soil
255 = ignore_index
````

Raw label mappings are sensor-specific and are defined in the sensor YAML files.

```yaml
# configs/sensors/spot.yaml
label_mapping:
  invalid_pixel: 0
  sealed_soil: 1
  non_sealed_soil: 2
```

```yaml
# configs/sensors/planetscope.yaml
label_mapping:
  invalid_pixel: 0
  sealed_soil: 1
  non_sealed_soil: 2
```

---

## 0. Expected Project Structure

```text
data/raw/
├── spot/
│   ├── images/
│   └── labels/
└── planetscope/
    ├── images/
    └── labels/

data/processed/
└── {model_name}/
    └── {sensor_setup}/
        ├── train/
        │   ├── images/
        │   └── labels/
        ├── val/
        │   ├── images/
        │   └── labels/
        └── test/
            ├── images/
            └── labels/

configs/
│
├── sensors/
│   ├── spot.yaml
│   └── planetscope.yaml
│
├── splits/
│   ├── raw/
│   │   ├── planetscope/
│   │   ├── spot/
│   │   └── mixed/
│   │
│   └── processed/
│       ├── crossearth/
│       ├── deeplabv3plus/
│       ├── segformer_sae/
│       └── dofa/
│
└── training/
    ├── crossearth/
    ├── deeplabv3plus/
    ├── segformer_sae/
    └── dofa/
```

Every raw image must have a corresponding label with the same file name.

Example:

```text
data/raw/planetscope/images/tile_001.tif
data/raw/planetscope/labels/tile_001.tif
```

Training configs follow this convention:

```text
configs/training/{model_name}/{sensor_setup}/online.yaml
configs/training/{model_name}/{sensor_setup}/offline.yaml
```

where:

```text
model_name    = crossearth | deeplabv3plus | segformer_sae | dofa
sensor_setup  = planetscope | spot | mixed
```

---

## 1. Raw Label Checks

Before training or preprocessing, check the raw label values.

### 1.1 Check PlanetScope Raw Labels

```powershell
python -c "import rasterio, numpy as np, glob, collections; files=glob.glob('data/raw/planetscope/labels/*.tif'); total=collections.Counter(); print('n files',len(files)); [total.update(dict(zip(*[a.tolist() for a in np.unique(rasterio.open(f).read(1), return_counts=True)]))) for f in files]; print(dict(sorted(total.items())))"
```

Expected PlanetScope raw labels:

```text
{0: ..., 1: ..., 2: ...}
```

### 1.2 Check SPOT Raw Labels

```powershell
python -c "import rasterio, numpy as np, glob, collections; files=glob.glob('data/raw/spot/labels/*.tif'); total=collections.Counter(); print('n files',len(files)); [total.update(dict(zip(*[a.tolist() for a in np.unique(rasterio.open(f).read(1), return_counts=True)]))) for f in files]; print(dict(sorted(total.items())))"
```

Expected SPOT raw labels:

```text
{0: ..., 1: ..., 3: ...}
```

---

## 2. Create Raw Splits

Raw split files point to the original GeoTIFF files.

Raw splits are model-independent and are stored under:

```text
configs/splits/raw/{sensor_setup}/
```

Generated sample format:

```yaml
- image: data/raw/planetscope/images/tile_001.tif
  label: data/raw/planetscope/labels/tile_001.tif
  sensor: planetscope
  sensor_config: configs/sensors/planetscope.yaml
```

### 2.1 Create PlanetScope Raw Split

```powershell
python scripts/create_splits.py --images data/raw/planetscope/images --labels data/raw/planetscope/labels --sensor planetscope --sensor-config configs/sensors/planetscope.yaml --out-dir configs/splits/raw/planetscope --train-ratio 0.70 --val-ratio 0.15 --test-ratio 0.15 --seed 42
```

Output:

```text
configs/splits/raw/planetscope/train_samples.yaml
configs/splits/raw/planetscope/val_samples.yaml
configs/splits/raw/planetscope/test_samples.yaml
configs/splits/raw/planetscope/split_summary.yaml
```

### 2.2 Create SPOT Raw Split

```powershell
python scripts/create_splits.py --images data/raw/spot/images --labels data/raw/spot/labels --sensor spot --sensor-config configs/sensors/spot.yaml --out-dir configs/splits/raw/spot --train-ratio 0.70 --val-ratio 0.15 --test-ratio 0.15 --seed 42
```

Output:

```text
configs/splits/raw/spot/train_samples.yaml
configs/splits/raw/spot/val_samples.yaml
configs/splits/raw/spot/test_samples.yaml
configs/splits/raw/spot/split_summary.yaml
```

## 2.3 Create Mixed SPOT + PlanetScope Raw Split

```powershell
python scripts/create_splits.py --dataset-root data/raw --sensors spot planetscope --sensor-config-root configs/sensors --out-dir configs/splits/raw/mixed --train-ratio 0.70 --val-ratio 0.15 --test-ratio 0.15 --seed 42
```

Output:

```text
configs/splits/raw/mixed/train_samples.yaml
configs/splits/raw/mixed/val_samples.yaml
configs/splits/raw/mixed/test_samples.yaml
configs/splits/raw/mixed/split_summary.yaml
```

---

# 3. Generate Training Configs

Training configs can be generated automatically for all models, all sensor setups and both modes.

The generator creates:

```text
configs/training/{model_name}/{sensor_setup}/online.yaml
configs/training/{model_name}/{sensor_setup}/offline.yaml
```

## 3.1 Generate All Configs

```powershell
python scripts/generate_training_configs.py
```

## 3.2 Overwrite Existing Configs

```powershell
python scripts/generate_training_configs.py --overwrite
```

## 3.3 Generate Only CrossEarth Configs

```powershell
python scripts/generate_training_configs.py --models crossearth --overwrite
```

## 3.4 Generate Only Mixed Configs

```powershell
python scripts/generate_training_configs.py --datasets mixed --overwrite
```

Expected output structure:

```text
configs/training/
├── crossearth/
│   ├── planetscope/
│   │   ├── online.yaml
│   │   └── offline.yaml
│   ├── spot/
│   │   ├── online.yaml
│   │   └── offline.yaml
│   └── mixed/
│       ├── online.yaml
│       └── offline.yaml
│
├── deeplabv3plus/
│   ├── planetscope/
│   │   ├── online.yaml
│   │   └── offline.yaml
│   ├── spot/
│   │   ├── online.yaml
│   │   └── offline.yaml
│   └── mixed/
│       ├── online.yaml
│       └── offline.yaml
│
├── segformer_sae/
│   ├── planetscope/
│   │   ├── online.yaml
│   │   └── offline.yaml
│   ├── spot/
│   │   ├── online.yaml
│   │   └── offline.yaml
│   └── mixed/
│       ├── online.yaml
│       └── offline.yaml
│
└── dofa/
    ├── planetscope/
    │   ├── online.yaml
    │   └── offline.yaml
    ├── spot/
    │   ├── online.yaml
    │   └── offline.yaml
    └── mixed/
        ├── online.yaml
        └── offline.yaml
```

---

# 4. Check Online Dataset Pipeline

These commands validate the online pipeline:

```text
Raw GeoTIFF
→ MultiSensorSegDataset
→ preprocess_image_for_model()
→ preprocess_label_for_model()
→ DataLoader
→ batch ready for training
```

For CrossEarth RGBNIR, the expected batch shape is:

```text
image shape = (B, 4, 504, 504)
label shape = (B, 504, 504)
```

For DeepLabV3+ and SegFormerSAE RGBNIR:

```text
image shape = (B, 4, 512, 512)
label shape = (B, 512, 512)
```

For DOFA:

```text
image shape = (B, C, 224, 224)
label shape = (B, 224, 224)
```

### Example: Check PlanetScope Train — CrossEarth

```powershell
python scripts/check_dataset.py --samples configs/splits/raw/planetscope/train_samples.yaml --model-name crossearth --split train --patch-size-px 512 --batch-size 4 --num-workers 0 --expected-channels 4 --expected-size 504 --max-batches 5
```

---

# 5. Online Training from Raw GeoTIFF

In this mode, the Dataset reads the raw GeoTIFF files and performs preprocessing online.

Advantages:

* maximum flexibility;
* random crops can change across epochs;
* useful for debugging.

Disadvantages:

* slower for large datasets.

Online configs are stored as:

```text
configs/training/{model_name}/{sensor_setup}/online.yaml
```

### Example: CrossEarth PlanetScope Online Fine-Tuning

```powershell
python train/train.py --config configs/training/crossearth/planetscope/online.yaml --device cuda:0 --train-mode finetune
```

### Example: CrossEarth PlanetScope Online Training from Scratch

```powershell
python train/train.py --config configs/training/crossearth/planetscope/online.yaml --device cuda:0 --train-mode scratch
```


---

# 6. Offline Preprocessing

For large datasets, offline preprocessing is recommended.

The offline pipeline is:

```text
Raw GeoTIFF
→ scripts/build_processed_dataset.py
→ data/processed/{model}/{sensor}/{split}/images/*.tif
→ data/processed/{model}/{sensor}/{split}/labels/*.tif
→ configs/splits/processed/{model}/{sensor}/*_samples.yaml
→ ProcessedSegDataset
→ DataLoader
→ training
```

Offline preprocessing performs:

* crop extraction;
* band selection;
* radiometric normalization;
* model-aware resize;
* label remapping.

Training-time augmentation remains online:

* horizontal / vertical flips;
* rotations;
* brightness;
* contrast;
* noise.

Processed splits are model-dependent and are stored under:

```text
configs/splits/processed/{model_name}/{sensor_setup}/
```

---

## 6.1 Build Processed Dataset — Example: CrossEarth PlanetScope

```powershell
python scripts/build_processed_dataset.py --config configs/training/crossearth/planetscope/offline.yaml --split train
python scripts/build_processed_dataset.py --config configs/training/crossearth/planetscope/offline.yaml --split val
python scripts/build_processed_dataset.py --config configs/training/crossearth/planetscope/offline.yaml --split test
```

---

# 7. Offline Training from Processed GeoTIFF

For offline training, the YAML config must use:

```yaml
data:
  preprocessed: true
  train_samples: configs/splits/processed/{model_name}/{sensor_setup}/train_samples.yaml
  val_samples: configs/splits/processed/{model_name}/{sensor_setup}/val_samples.yaml
```

In this mode, `ProcessedSegDataset` is used instead of `MultiSensorSegDataset`.

Offline configs are stored as:

```text
configs/training/{model_name}/{sensor_setup}/offline.yaml
```

### Example: CrossEarth PlanetScope Offline Fine-Tuning

```powershell
python train/train.py --config configs/training/crossearth/planetscope/offline.yaml --device cuda:0 --train-mode finetune
```

### Example: PlanetScope Offline Training from Scratch

```powershell
python train/train.py --config configs/training/crossearth/planetscope/offline.yaml --device cuda:0 --train-mode scratch
```

---

# 8. Inference

Use the best checkpoint, for example:

```text
outputs/checkpoints/crossearth_finetune/best.pth
```

or:

```text
outputs/checkpoints/crossearth_scratch/best.pth
```

### Example: Single Prediction - PlanetScope with CrossEarth

```powershell
python inference/predict_tile.py --image data/raw/planetscope/images/S4-02-01a-VA1_train_autumn_gt-012-015_512_7680.tif --sensor-config configs/sensors/planetscope.yaml --checkpoint outputs/checkpoints/crossearth_finetune/best.pth --output outputs/predictions/crossearth_planetscope_pred.tif --device cuda:0 --output-mode qgis --amp
```

### Example: Batch Prediction - PlanetScope with CrossEarth

```powershell
python inference/predict_folder.py --model crossearth --input-dir data/raw/planetscope/images --sensor-config configs/sensors/planetscope.yaml --checkpoint outputs/checkpoints/crossearth_finetune/best.pth --output-dir outputs/predictions/crossearth/planetscope --device cuda:0 --recursive --skip-existing
```
---