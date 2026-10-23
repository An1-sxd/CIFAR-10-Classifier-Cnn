# CIFAR-10 Image Classifier with PyTorch

A convolutional neural network (CNN) for classifying RGB images from the
CIFAR-10 dataset using PyTorch and Torchvision.

## Project overview

The notebook:

1. Downloads CIFAR-10.
2. Applies data augmentation to the training images.
3. Normalizes the images using the CIFAR-10 channel statistics used in the
   notebook.
4. Splits the 50,000-image training set into training and validation subsets.
5. Trains a CNN while tracking training and validation loss.
6. Restores the checkpoint with the lowest validation loss.
7. Evaluates the restored model on the held-out test set.

## Results

The recorded notebook run reports the following test-set results:

| Metric | Score |
| --- | ---: |
| Test loss | 0.4342 |
| Accuracy | 85.83% |
| Macro F1 score | 0.8593 |
| Micro F1 score | 0.8583 |
| Weighted F1 score | 0.8593 |

Results can vary depending on the PyTorch version, hardware, and random state.

## Dataset

CIFAR-10 contains 60,000 32 x 32 colour images across 10 classes:

1. airplane
2. automobile
3. bird
4. cat
5. deer
6. dog
7. frog
8. horse
9. ship
10. truck

The notebook uses:

- 45,000 images for training
- 5,000 images for validation
- 10,000 images for testing
- A fixed seed of `42` for the train-validation split
- RGB input tensors with shape `3 x 32 x 32`

## Preprocessing and augmentation

Training images use random cropping and horizontal flipping before
normalization:

```python
transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616),
    ),
])
```

Validation and test images are converted to tensors and normalized without
random augmentation.

## Model architecture

The model is implemented as a custom `nn.Module` named `Cnn`:

```text
Input: 3 x 32 x 32
  Conv2d(3, 32, kernel_size=3, padding=1)
  BatchNorm2d(32)
  ELU
  MaxPool2d(2)
  Conv2d(32, 64, kernel_size=3, padding=1)
  BatchNorm2d(64)
  ELU
  MaxPool2d(2)
  Conv2d(64, 128, kernel_size=3, padding=1)
  BatchNorm2d(128)
  ELU
  MaxPool2d(2)
  Conv2d(128, 256, kernel_size=3, padding=1)
  BatchNorm2d(256)
  ELU
  MaxPool2d(2)
  Flatten
  Linear(256 * 2 * 2, 128)
  BatchNorm1d(128)
  ReLU
  Linear(128, 10)
Output: class logits for the 10 CIFAR-10 classes
```

## Training configuration

- Loss function: `CrossEntropyLoss`
- Optimizer: `Adam`
- Learning rate: `0.001`
- Weight decay: `0.0001`
- Epochs: `70`
- Batch size: `64`
- Device: CUDA when available, otherwise CPU
- Best checkpoint: model state with the lowest validation loss

## Evaluation

The notebook evaluates the restored checkpoint using:

- Multiclass accuracy
- Confusion matrix
- Macro F1 score
- Micro F1 score
- Weighted F1 score

The confusion matrix is calculated in the class order listed in the Dataset
section.

## Requirements

- Python 3.10+
- PyTorch
- Torchvision
- TorchMetrics
- NumPy
- Matplotlib
- Jupyter Notebook or JupyterLab

Install the dependencies with:

```bash
pip install torch torchvision torchmetrics numpy matplotlib jupyter
```

## Running the notebook

1. Open `notebook.ipynb` in Jupyter or Google Colab.
2. Run the cells from top to bottom.
3. If running in Google Colab, mount Google Drive when prompted. The notebook
   stores the dataset under:

   ```text
   /content/drive/MyDrive/Cifar-10
   ```

4. If running locally, replace the Google Drive dataset path with a local
   directory before executing the dataset-loading cells.

## Project files

- `notebook.ipynb` - data loading, exploratory visualization, model definition,
  training, checkpoint selection, and evaluation
- `readme.md` - project documentation
