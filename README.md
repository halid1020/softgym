<h1>  SoftGym: Extension on SoftGym for cloth-shaping benchmarks and oracles </h1>

This fork is extended on the original [`SoftGym`](https://github.com/Xingyu-Lin/softgym) with modifications mainly on the cloth environments --- Note that the other original environments of `SoftGym` do not work properly in this version. If there is a conflict of interest, please contact `ah390@st-andrews.ac.uk `. This fork supports benchmark environments `mono-square-fabric `, `rainbow-square-fabrics `, `rainbow-rectangular-fabrics `, `realadapt-towels, ` `real2sim-towels-sq, clothfunnels-realadapt-<garment> `; These benchmarks used by `PlaNet-ClothPick `[3],`JA-TN `[2] and `DRAPER` [1] projects.

This repository is authored by Halid Abdulrahim Kadi and supervised by Kasim Terzić; Luis Figueredo and Praminda Caleb-Solly provided some insights for `realadapt` benchmark environments; and, Ryan Haward provided some contributions to this `README` file

## I. Install and Setup the Simulator

1. Install relevant packages before compiling the simulation.

If you want to run with display mode, please ensure to install OpenGL-related packages into your machine, e.g.,

```
apt-get install build-essential libgl1-mesa-dev freeglut3-dev libglfw3 libgles2-mesa-dev
```

2. Ensure you have downloaded and installed `anaconda3` right under your home directory regarding your operating system version; you can do so by following the [tutorial](https://docs.anaconda.com/free/anaconda/install/linux/).
3. Then, create conda environment for compiling and running the simulation by following the bellow commands under the root directory of the `softgym` (after `git clone` it to your machine):

```
conda env create -f environment.yml
```

Note that if you want to remove the environment

```
conda remove -n softgym --all  
```

4. Download and install the [`cloth_initial_states.zip`](https://drive.google.com/file/d/1bFgrjLffy9q4PIWHGfFRCHGEzSo8iHfE/view?usp=sharing). Note that you can skip this step if you want the environments themselves generates the corresponding initial states automatically at the begining of initialisation, but it may take quite a long time.

```

# Do not forget to install `gdown` using `pip install gdown`.

gdown https://drive.google.com/uc?id=1c6vPb-TVqkqOkc5-X33nDVJm5e2LPUG7

mv cloth_initial_states.zip <path_to_softgym>/softgym

cd <path_to_softgym>/softgym 

mkdir cached_initial_states

unzip cloth_initial_states.zip && mv cloth_initial_states/*.pkl cached_initial_states/
```

5. Ensure `nvidia-docker` is installed (this is deprecated), as we need to use docker environment to compile the simulation environment; it can be installed by following this [tutorial](https://docs.docker.com/engine/install/ubuntu/) and this [tutorial](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/1.10.0/install-guide.html).
6. Compile the simulator inside a docker file provided by the original authors of `SoftGym`.

```
nvidia-docker run -v <path_to_softgym>/softgym:/workspace/softgym \
-v $HOME/anaconda3:$HOME/anaconda3 \
-it xingyu/softgym:latest bash

# Or (recommend)

docker run -v <path_to_softgym>/softgym:/workspace/softgym \
-v $HOME/anaconda3:$HOME/anaconda3 \
-it xingyu/softgym:latest bash

```

Compile the simulator in the docker:

```
export PATH="<absolute_path_to_home_dir>/anaconda3/bin:$PATH"

cd softgym

. ./setup.sh  && . ./compile.sh
```

Note that the <absolute_path_to_home_dir> should be the `$HOME` from **OUTSIDE** the docker, not from inside it (`$HOME` inside the docker is /root/, which isn't where we've mapped `anaconda3`).

# II. Run Oracle Policies

You do not need to employ the docker container used during the compilation in this section, but you do need to do the setup again under the root directory of the repository.

```
. ./setup.sh
```

Then, you can continue the following instructions right under the root directory.

## A. Flattening Oracles

We support two smoothing oracle policies `oracle-towel-smoothing` and `realadapt-OTS`. For example, run `realadapt-OTS` oracle policy in `realadapt-towels`:

```
python run.py --domain realadapt-towels --initial crumpled \
    --task flattening --policy realadapt-OTS --eid 1 --save_video
```

## B. Folding oracles

The supported folding types include `one-corner-inward-folding`, `double-corner-inward-folding`, `all-corner-inward-folding`, `diagonal-folding`, `digonal-cross-folding`, `corners-edge-inward-folding`, `rectangular-folding`, `side-folding` and `double-side-folding`. Note that some folding types are only supported in square-fabric benchmark environments.

For example, run `all corner-inward folding` in `realadapt-towels` from `flattened` initial positions.

```
python run.py --domain realadapt-towels-sq --initial flattened  \
    --task all-corner-inward-folding --policy all-corner-inward-folding \
    --eid 0 --save_video



```

# Related Papers

[1] Kadi HA, Chandy JA, Figuerdo L, Terzić K, Caleb-Solly P. DeepCloth-ROB2QSP&P: Towards a Robust Robot Deployment for Quasi-Static Pick-and-Place Cloth-Shaping Neural Controllers. arXiv preprint arXiv:2409.15159 2024.

[2] Kadi HA, Terzić K. JA-TN: Pick-and-Place Towel Shaping from Crumpled States based on TransporterNet with Joint-Probability Action Inference. In8th Annual Conference on Robot Learning 2024.

[3] Kadi HA, Terzić K. PlaNet-ClothPick: effective fabric flattening based on latent dynamic planning. In2024 IEEE/SICE International Symposium on System Integration (SII) 2024 Jan 8 (pp. 972-979). IEEE.
