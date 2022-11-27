# HPC Virtual Env Guide
1. Load module `module load python3/3.9.14`
2. `python3 -m venv venv`
3. `source venv/bin/activate`
4. Install requirements:  `pip install -r requirements.txt`
5. Install CUDA-compatible torch versions: `pip3 install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html`

# Bachelorproject
A repo for the bachelor project for the BSc in Artificial Intelligence and Data, by Jonas Poulsen, Lasse Møller Sørensen and Christian Westerdahl in collaboration with WSAudiology.

## ToDo
### First stage
- [X] `Save output data in a smart manners...........(Jonas)`
  - [X] `Organize docs folders.......................(Jonas)`
- [X] `Implement hyperparameter-optimization.........(Jonas)`
    - [X] `Move model-init to model_tools.py from framework`
- [X] `Implement Filterbank (Pytorch Torchaudio).....(Christian)`
- [X] `Update CNN....................................(Christian)`
    - [X] `Changing EfficientNet for MobileNet`
- [X] `Make sure uncertiany plots work...............(Jonas)`
  - [X] `Revisit these and look at background imggrid(Jonas)`
- [X] `Implement VAAL................................(Lasse and Christian)`
- [X] `Implement Hydra...............................(Jonas)`
- [X] `Implement 'save model' function...............(Jonas)`
- [X] `Uncertainty sampling giving NAN values........(Jonas)`
- [X] `Make simple and VAAL combatible...............(Jonas)`
- [ ] `Preprocess WSA data...........................(Jonas)`
- [X] `Fix efficienet for GPU training...............(Lasse)`
- [ ] `1. HPC run: Simple model, CNN, Random, Uncertainty and VAAL (Lasse)`

### Second stage
- [ ] `HEAR-DS dataset implementation`


### WSA stage
- [ ] `Label subdataset`
  - [X] `Implement easy program for labeling (Streamlit or penguin) (Jonas)`
- [ ] `Get results `
- [ ] `Awesome but optional: Build full labelling framework with running active learning (streamlit?)`

### Not urgent
- [ ] `Change final plot two take different models and tasklearner`
    - [ ] `Bars instad of transparent area`
- [ ] `Clean up old files and functions and restructure`

### Optional
- [ ] `Eval. metrics: mean Average precision ect.....(Jonas)`
  - [ ] `Implement one-hot encoding https://pytorch.org/docs/stable/generated/torch.nn.functional.one_hot.html` 
  - [ ] `Use torch F.binary_cross_entropy as loss https://pytorch.org/docs/stable/generated/torch.nn.functional.binary_cross_entropy.html`
- [ ] `FSD50K` 
    - [ ] `Overcome the multiclass problem`
    - [ ] `Variable sized input`
