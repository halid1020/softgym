<h1>  SoftGym V2: Extension on SoftGym for PlaNet-ClothPick </h1>

Authored by Halid Abdulrahim Kadi and supervised by Kasim Terzic; Ryan Haward also made contribution to this README file.

This version of the `SoftGym` extended on the orignal `SoftGym` with modification minaly on the cloth environments. The other environments does not work properly in this version.


# Install and Setup the Simulator


1. Install relavent packages before compiling the simulation.

If you want to run the simulation with its display mode, please ensure to install OpenGL related packages into your machine, e.g.,
```
apt-get install build-essential libgl1-mesa-dev freeglut3-dev libglfw3 libgles2-mesa-dev
```

Please also ensure `anaconda` is installed under your home directory.


Then, create conda environment for compiling and running the simulation by running the follow command under the root directory of the repository:


```
conda env create -f environment.yml
```

Note that if you want to remove the environment

```
conda remove -n softgym --all  
```

2. Download `mono_square_fabric.pkl` intial-state data file for square-fabric environment.

TODO: we need to provide a download link and command instructions.


3. You will need to compile the simulator inside a provided docker by the original authors of `SoftGym`.

Download and inititiate the provided image:

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

4. After compiling the simulator, please exit the docker.
