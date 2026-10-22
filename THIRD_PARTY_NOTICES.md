# Third-party components

The native integration imports VACE and its Wan video components from a separately configured installation. Upstream source and model weights are not bundled in this repository.

- VACE: https://github.com/ali-vilab/VACE
- Wan2.1: https://github.com/Wan-Video/Wan2.1
- PyTorch, NumPy, OpenCV and Pillow are external dependencies.
- Archived visual feature extraction additionally references torchvision ResNet18 and Hugging Face SigLIP2/Transformers. These packages/weights are not required for numeric replay and are not bundled.

Retain applicable upstream notices and terms when obtaining these components. This repository does not relicense upstream source or model weights. A project-wide distribution license for this research snapshot has not yet been selected.
