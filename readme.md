# MobileGaze

## Requirements
Ubuntu 20.04(WSL)  

pytorch 2.5.1

WSLs can only be used on servers that use Nvidia GeForce graphics cards, as WSLs can only call graphics cards that HyperV can recognize.

## Usage
### Prepare Conda Environment
```bash
    git clone https://github.com/NethengeicWE/MobileGaze
    conda create -n mamba python=3.10
    conda activate mamba
    pip install torch==2.5.1+cu118 --index-url https://download.pytorch.org/whl/cu121
    pip install -r requirement.txt
```
### Prepare Dataset
You need to prepare original datasets and preprocess them by <a href="http://phi-ai.org/GazeHub/" target="_blank">*Phi-ai Lab.*</a>. Original datset links are in Links.   
```bash
    mkdir ~/MobileGaze/dataset
    mv Preprocessed_dataset  ~/MobileGaze/dataset
```
You can run `python dataset_xxx_Gazehub_eyeplus.py` to check if the dataset can be loaded correctly.

For ETH-XGaze dataset, we provide a script to abstract eye crop from face image by mediapipe.
```bash
    python ethx_eyecrop_extract.py
```
It will take few hours to finish because this dataset is very big. 

> Note: Due to the extreme head posture of some samples, about 5% of the cropping results are incorrect, which is a normal phenomenon.

### Prepare pretrain module
We load the model by offline weights(Because I cannot download by timm), you need to download the model weight file on Hugging face and move it to the folder.
```bash
   mkdir -p ~/MobileGaze/model_use/mobilevit_xs
   wget -O ~/MobileGaze/model_use/mobilevit_xs/model.safetensors https://huggingface.co/timm/mobilevit_xs.cvnets_in1k/resolve/main/model.safetensors
```
### Config rootfile
Replace as config.py follow：
```
# old
/home/NWE/miniconda3/envs/worksacpe_gaze 
# Yours
~/MobileGaze
```
### Run
* Star Train: 
```python
python train_GAS.py
```
> To do the LOSO and other experiments: see `__main__` annotation function.

* Validate and test
Integerate in `train_GAS.py`, but you can use `attentionHeatMap.py` to check sample result, compare the model's attention area and prediction results with the ground truth. 

## Link of origin dataset





