# Machine Learning Nanodegree Capstone

Read the report [here](https://domluna.me/project/ml_capstone/).

### Setup

There are two ways to run the code.

1. Install and run the docker image. The image is pretty beefy ~1.5GB uncompressed and the doom environments take a few minutes to install. I haven't yet figured out how to display the video of an episode through the container, so no video will be shown but  total reward and timesteps taken will be printed.
2. Install the dependencies for the project on your local machine.

The only thing that might be tricky to install on the local machine would be the Doom environments so I would try that first and then resort to the docker image.

#### Docker

```sh
docker pull domluna/ml-capstone
docker run -it --rm domluna/ml-capstone /bin/bash
```

Once you're in the container

```sh
cd ml_p5_capstone
```

From here you can run the files described below.

#### Local

```sh
pip install gym[doom] theano keras tabulate numpy scipy scikit-image matplotlib h5py
git clone https://github.com/domluna/modular_rl.git
cd modular_rl
git checkout doms-branch

# Finally add modular_rl to your PYTHONPATH
export PYTHONPATH=/path/to/modular_rl:$PYTHONPATH
```

### Run files

The main files are `train.py` and `play.py`. The former being for training an agent and the later for evaluating by playing episodes. Training a model takes a few hours.


Here we train a Feedforward TRPO agent on the DoomHealthGathering-v0 environment and save snapshots every 10 iterations.

```sh
KERAS_BACKEND=theano python train.py --gamma=0.995 --lam=0.97 --agent=modular_rl.agentzoo.TrpoAgent --max_kl=0.01 --cg_damping=0.1 --activation=tanh --n_iter=250 --seed=0 --timesteps_per_batch=5000 --env=DoomHealthGathering-v0 --outfile=$HOME/rl_results/DoomHealthGathering1.h5 --use_hdf 1 --snapshot_every 10
```

Playing example:

We load the model we trained and play 5 episodes.

```sh
KERAS_BACKEND=theano python play.py --agent=modular_rl.agentzoo.TrpoAgent --episodes=5 --env=DoomHealthGathering-v0 --load_snapshot=$HOME/rl_results/DoomHealthGathering1.h5
```

### Run saved snapshot models

Run models from the `snapshots` directory.

```sh
KERAS_BACKEND=theano python play.py --agent=modular_rl.agentzoo.TrpoAgentCNN --episodes=5 --env=DoomCorridor-v0 --load_snapshot=snapshots/DoomCorridor-CNN.h5

KERAS_BACKEND=theano python play.py --agent=modular_rl.agentzoo.TrpoAgent --episodes=5 --env=DoomHealthGathering-v0 --load_snapshot=snapshots/DoomHealthGathering-FF.h5

KERAS_BACKEND=theano python play.py --agent=modular_rl.agentzoo.TrpoAgentCNN --episodes=5 --env=DoomHealthGathering-v0 --load_snapshot=snapshots/DoomHealthGathering-CNN.h5
```

