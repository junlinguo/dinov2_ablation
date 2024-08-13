# Project Notes

## Dataset
### imagenet1k download 
```bash
    wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar --no-check-certificate
    
    wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar --no-check-certificate
```
```bash
    # Extract train data 
    
    mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train
    tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
    find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}";     tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
    cd ..

```
the bash script ./valprep.sh can be found [here](https://github.com/PatrickHua/EasyImageNet/blob/main/valprep.sh)
```bash
    # Extract val data 
    mkdir val && mv ILSVRC2012_img_val.tar val/ 
    mv valprep.sh val/ && cd val && tar -xvf ILSVRC2012_img_val.tar
    ./valprep.sh

```

### imagenet1k data 
A github repo explains the data preparation: 

https://github.com/QTIM-Lab/dinov2_imagenet1k_example?tab=readme-ov-file

The root directory of the dataset should hold the following contents:

- `<ROOT>/test/ILSVRC2012_test_00000001.JPEG`
- `<ROOT>/test/[..]`
- `<ROOT>/test/ILSVRC2012_test_00100000.JPEG`
- `<ROOT>/train/n01440764/n01440764_10026.JPEG`
- `<ROOT>/train/[...]`
- `<ROOT>/train/n15075141/n15075141_9993.JPEG`
- `<ROOT>/val/n01440764/ILSVRC2012_val_00000293.JPEG`
- `<ROOT>/val/[...]`
- `<ROOT>/val/n15075141/ILSVRC2012_val_00049174.JPEG`
- `<ROOT>/labels.txt`

The provided dataset implementation expects a few additional metadata files to be present under the extra directory:
- `<EXTRA>/class-ids-TRAIN.npy`
- `<EXTRA>/class-ids-VAL.npy`
- `<EXTRA>/class-names-TRAIN.npy`
- `<EXTRA>/class-names-VAL.npy`
- `<EXTRA>/entries-TEST.npy`
- `<EXTRA>/entries-TRAIN.npy`
- `<EXTRA>/entries-VAL.npy`

These metadata files can be generated (once) with the following lines of Python code:
```python
from dinov2.data.datasets import ImageNet

for split in ImageNet.Split:
    dataset = ImageNet(split=split, root="<ROOT>", extra="<EXTRA>")
    dataset.dump_extra()
```

For my github repo, the code can be found as `imagenet1k-setup`:
the code has some flag variables. `setup` is to getting the extra/meta files ready 
```bash
    setup - Generate extra files for the ImageNet dataset
    debug - Test the setup by loading a sample and checking the output
    customized - The customized dataset class and augmentations
```
E.g., Define the root paths to the ImageNet dataset and the directory for extra metadata files
```python
root = '/home/guoj5/Documents/datasets/imagenet1k'
extra = '/home/guoj5/Documents/datasets/imagenet1k_meta'

```
The paths defined in dgx server: 
    
    root = /workspace/data/preprocessing/imagenet1k_download_new/root
    
    extra = /workspace/data/preprocessing/imagenet1k_download_new/extra
    
## SSL architecture 
- ssl arch: /dinov2/train/ssl_meta_arch.py. Student models, Teacher model  logic 
- train script: /dinov2/train/trainxx.py

### Model 
This vit model can be changed for different input shape, for example, the in chans can vary. 
if rgb: modify this /dinov2/models/vision_transformer.py. in_chan

