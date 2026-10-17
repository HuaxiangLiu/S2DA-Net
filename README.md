# S2DA-Net

This code is for the paper title 'S2DA-Net: Spatial and spectral-learning double-branch aggregation network for liver tumor segmentation in CT images' 


# dataset 
LiTS2017：https://competitions.codalab.org/competitions/17094#participate

3D-IRCADb 01：[https://www.ircad.fr/research/3d-ircadb-01/](https://www.ircad.fr/research/data-sets/liver-segmentation-3d-ircadb-01/)


# How to use this code

1. Download nnUNet framework from the website:https://github.com/MIC-DKFZ/nnUNet, and use the following commands:

git clone https://github.com/MIC-DKFZ/nnUNet.git (note: nnUNetv1 version on NVDIO A6000 GPU)

cd nnUNet
pip install -e .

Noting: you must use the Apex

2. The above files including SpectralNet_S2DA.py is put into the fold: nnunetv2\training\nnUNetTrainer\variants\network_architecture.  Then, you should unzip the file 'fft.zip', and input the fold named 'fft' with the equal location of SpectralNet_S2DA.py.

3. run the command 'python run_training.py'.

If you encounter any problems or need the pretaied model, please do not hesitate to contact us.

# use this code,please cite this paper



H. Liu, J. Yang, C. Jiang, et al. S2DA-Net: Spatial and spectral-learning double-branch aggregation network for liver tumor segmentation in CT images, Computers in Biology and Medicine, Volume 174, 108400, 2024. doi: [https://doi.org/10.1016/j.compbiomed.2022.106352](https://doi.org/10.1016/j.compbiomed.2024.108400)
