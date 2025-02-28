The following guidance works with 2080Ti GPU, cuda 11.7, driver 515.65.01.

git clone this repo and `cd` into it.

    git clone https://github.com/zql-kk/FlowPolicy.git

---

1.create python env

    conda create -n flowpolicy python=3.8
    conda activate flowpolicy

---

2.install torch

    # cuda=11.7
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
    # Note: just install the torch version that matches your own cuda version

---

3.install flowpolicy

    cd FlowPolicy && pip install -e . && cd ..

---

4.install mujoco in `~/.mujoco`

    cd ~/.mujoco
    wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz -O mujoco210.tar.gz --no-check-certificate

    tar -xvzf mujoco210.tar.gz
---

5.put the following into your bash script (usually in `YOUR_HOME_PATH/.bashrc`). Remember to `source ~/.bashrc` to make it work and then open a new terminal.

    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${HOME}/.mujoco/mujoco210/bin
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
    export MUJOCO_GL=egl

---

6.install mujoco-py (in the folder of `third_party`):

    cd YOUR_PATH_TO_THIRD_PARTY
    cd mujoco-py-2.1.2.14
    pip install -e .
    cd ../..

----

7.install sim env

    pip install setuptools==59.5.0 Cython==0.29.35 patchelf==0.17.2.0

    cd third_party
    cd gym-0.21.0 && pip install -e . && cd ..
    cd Metaworld && pip install -e . && cd ..
    cd rrl-dependencies && pip install -e mj_envs/. && pip install -e mjrl/. && cd ..

---

8.install pytorch3d

    conda install -y https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch3d/linux-64/pytorch3d-0.7.5-py38_cu117_pyt201.tar.bz2
    or (like DP3)
    cd third_party/pytorch3d_simplified && pip install -e . && cd ..

---

9.other necessary packages

    pip install zarr==2.12.0 wandb ipdb gpustat dm_control omegaconf hydra-core==1.2.0 dill==0.3.5.1 einops==0.4.1 diffusers==0.11.1 numba==0.56.4 moviepy imageio av matplotlib termcolor natsort open3d

---

10.install the visualizer for pointclouds offered by DP3 (optional)

    pip install kaleido plotly
    cd visualizer && pip install -e . && cd ..