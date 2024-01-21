# AI (ResNet-50) in Micro-CT Defect Detection: A SiC Boule Case Study
## How to run
### Dataset Structure
```
dataset_path
│
│
├── train
│   ├── carbon inclusion
│   │   └── .bmp files
│   ├── micropipe
│   │   └── .bmp files
│   ├── ...
│   └── single crystal
│       └── .bmp files
│
└── test
    ├── carbon inclusion
    │   └── .bmp files
    ├── micropipe
    │   └── .bmp files
    ├── ...
    └── single crystal
        └── .bmp files
```

### Training prerequisites
1. Prepare your dataset
2. Create environment and install dependencies:

``pip install tensorflow==2.4.0``
### Training
``python main.py --dataset_path ./your own path --phase train --save_path ./result
``
### Test
``
python main.py --dataset_path ./your own path --phase eval --inference_image_path ./test.bmp --save_path ./result
``
### Contact
For any inquiry please contact us at our email addresses: 19820220157165@stu.xmu.edu.cn