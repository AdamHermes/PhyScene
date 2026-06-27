wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh  

source ~/.bashrc
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
#### Create a Conda Environment
```
conda create -n physcene python=3.10 -y
conda activate physcene
```
#### Install Python Packages
```
#conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
```
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1

pip install \
numpy networkx pyrr pyyaml pillow scipy tqdm \
trimesh==3.21.7 matplotlib simple-3dviz==0.7.0 \
num2words nltk transformers==4.25.1 \
clip==0.2.0 seaborn wandb hydra-core tensorboard \
omegaconf loguru open3d==0.17.0 \
opencv-python==4.8.0.74 tabulate==0.9.0 \
einops smplx

pip install kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.1_cu121.html


python scripts/eval/evaluate_echoscene_collision.py \
    --object-mesh-root "/content/drive/MyDrive/EchoScene/col5/2050/echoscene/object_meshes" \
    --scene-json "/content/drive/MyDrive/EchoScene/col5/2050/merged.json" \
    --transform-source json
#### Compile Rotated_IoU Extension:   
```
# code is from https://github.com/lilanxiao/Rotated_IoU
cd models/loss/cuda_op
python setup.py install
```

#### Install ChamferDistancePytorch
```
cd ChamferDistancePytorch/chamfer3D
python setup.py install
```

#### Install Kaolin:
```
pip install kaolin==0.14.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.0.1_cu117.html
```

#### Modify simple_3dviz
The assets have materials with informal uv map, which will cause error when loading mesh in simple_3dviz.
(1) You may need to modify a command line in  ```~/anaconda3/envs/physcene/lib/python3.8/site-packages/simple_3dviz/io/multi_mesh.py``` at ```line 153``` from:
```
except IndexError:
    face_uv = np.zeros((len(face_vertices), 2))
```
to
```
except:
    face_uv = np.zeros((len(face_vertices), 2))
```

(2) You may also need to modify a command line in  ```~/anaconda3/envs/physcene/lib/python3.8/site-packages/simple_3dviz/io/material.py``` at ```line 151``` from:
```
elif l.startswith("illum"):
    material["mode"] = {
        "0" : "constant",
        "1" : "diffuse",
        "2" : "specular"
    }[l.split()[1]]
```
to
```
elif l.startswith("illum"):
    modelst = {"0" : "constant",
                "1" : "diffuse",
                "2" : "specular"
                }
    if l.split()[1] in modelst:
        material["mode"] = modelst[l.split()[1]]
    else:
        material["mode"] = "specular"
```
