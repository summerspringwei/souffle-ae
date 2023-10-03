## Install tensorflow==1.14.0
First install a anaconda with python3.7
```shell
wget https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh
```
Install anaconda:
```shell
bash Anaconda3-2019.03-Linux-x86_64.sh -b -p ~/anaconda3.7
```
Create a virtual env and install tensorflow==1.14.0
```shell
pip install https://files.pythonhosted.org/packages/f4/28/96efba1a516cdacc2e2d6d081f699c001d414cc8ca3250e6d59ae657eb2b/tensorflow-1.14.0-cp37-cp37m-manylinux1_x86_64.whl
```
Install the suitable deps
```shell
pip uninstall h5py
pip install  gast==0.2.2
```
Done!