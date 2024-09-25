<h1>  SoftGym (fit DeepCloth-ROB2QSP&P): Extension on SoftGym for towel-shaping benchmarks and oracles </h1>

Authored by Halid Abdulrahim Kadi and supervised by Kasim Terzic; Ryan Haward also contributed to this `README` file; Luis Figueredo and Praminda Caleb-Solly provided insights for `real2sim` benchmark environments.

This fork is extended on the original [`SoftGym`](https://github.com/Xingyu-Lin/softgym) with modification mainly on the cloth environments (Note that the other environments do not work properly in this version).

This fork of SoftGym supports benchmark environment `mono-square-fabric`, `rainbow-square-fabrics`, `rainbow-rectangular-fabrics`, `real2sim-towels` and `real2sim-towels-sq`; These benchmarks used by `PlaNet-ClothPick`, `JA-TN` and `DeepCloth-ROB2QSP&P` projects.


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

4. Download and install the [`cloth_initial_states.zip`](https://drive.google.com/file/d/1yPsEX1WikWO9RlWhkap7II9Xs0HIPj88/view?usp=sharing).
```

# Do not forget to install `gdown` using `pip install gdown`.

gdown https://drive.google.com/uc?id=1yPsEX1WikWO9RlWhkap7II9Xs0HIPj88 

mv cloth_initial_states.zip <path_to_softgym>/softgym/cached_initial_states

cd <path_to_softgym>/softgym/cached_initial_states

unzip cloth_initial_states.zip && mv cloth_initial_states/*.pkl .
```

5. Ensure `nvidia-docker` is installed, as we need to use docker environment to compile the simulation environment; it can be installed by following this [tutorial](https://docs.docker.com/engine/install/ubuntu/) and this [tutorial](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/1.10.0/install-guide.html).


6. Compile the simulator inside a docker file provided by the original authors of `SoftGym`.

```
nvidia-docker run -v <path_to_softgym>/softgym:/workspace/softgym \
-v $HOME/anaconda3:$HOME/anaconda3 \
-it xingyu/softgym:latest bash

```

Compile the simulator in the docker: 
```
export PATH="<absolute_path_to_home_dir>/anaconda3/bin:$PATH"

cd softgym

. ./setup.sh  && . ./compile.sh
```

Note that the <absolute_path_to_home_dir> should be the $HOME from OUTSIDE the docker, not from inside it ($HOME inside the docker is /root/, which isn't where we've mapped anaconda3.)

# II. Test Softgym

1. Ensure [`agent-arena`](https://github.com/halid1020/agent-arena) is installed.

2. Go to the directory of `softgym` and run `. ./setup.sh`.

3. Then, go to the direcotory of `agent-arena`, then run the following commands
```
. ./setup.sh

cd src/test

python test_arena --arena "softgym|domain:mono-square-fabric,initial:crumple,action:pixel-pick-and-place(1),task:flattening"
```


# III. Run oracle policies



## A. Flattening oracles

We support two smoothing oracle policies `oracle-towel-smoothing` and `real2sim-smoothing`.

For example, run `real2sim-smoothing` oracle policy in `real2sim-towels`
```
python run.py --domain real2sim-towels --initial crumpled --task flattening --policy real2sim-smoothing
```
## B. folding oracles

Run the follow command to run and evaluate the folding oracle policies
```
python run.py --domain <domain-name> --initial <crumpled/flattened> --task <folding-type> --policy <folding-type> --eid <episode-id>
```

Supported folding types `one-corner-inward-folding`, `double-corner-inward-folding`, `all-corner-inward-folding`,  `diagonal-folding`, `digonal-cross-folding`, `corners-edge-inward-folding`, `rectangular-folding`, `side-folding` and `double-side-folding`.


For example, run `all corner-inward folding` in `real2sim-towels` from `flattened` initial positions.
```
python run.py --domain real2sim-towels-sq --initial flattened --task all-corner-inward-folding --policy all-corner-inward-folding --eid 1
```

