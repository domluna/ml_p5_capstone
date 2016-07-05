"""
This script runs a policy gradient algorithm
"""

from gym.envs import make
from modular_rl import *
import argparse, sys, cPickle
from tabulate import tabulate
import shutil, os, logging
import gym
import numpy as np
from doom_py import ScreenResolution

from skimage.color import rgb2gray
from skimage.transform import resize


class ObFilter(object):
    def __init__(self, new_width, new_height):
        self.w = new_width
        self.h = new_height

    def __call__(self, ob):
        out = resize(rgb2gray(ob), (self.h, self.w))
        return out.reshape(out.shape + (1,))

    def output_shape(self, input_shape):
        return (self.h, self.w, 1)

class ActFilter(object):
    def __init__(self, lookup):
        self.lookup = lookup
        self.n = len(self.lookup)

    def __call__(self, act):
        action_list = np.zeros(43) # Doom has 43 actions
        action_list[self.lookup[act]] = 1
        return action_list

    def output_shape(self):
        return self.n


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    update_argument_parser(parser, GENERAL_OPTIONS)
    parser.add_argument("--env", required=True)
    parser.add_argument("--agent", required=True)
    parser.add_argument("--plot", action="store_true")
    args, _ = parser.parse_known_args([arg for arg in sys.argv[1:] if arg not in ('-h', '--help')])

    env = make(args.env)
    env.configure(screen_resolution=ScreenResolution.RES_160X120)
    env_spec = env.spec

    mondir = args.outfile + ".dir"
    if os.path.exists(mondir): shutil.rmtree(mondir)
    os.mkdir(mondir)
    env.monitor.start(mondir, video_callable=None if args.video else VIDEO_NEVER)
    agent_ctor = get_agent_cls(args.agent)
    update_argument_parser(parser, agent_ctor.options)
    args = parser.parse_args()
    if args.timestep_limit == 0:
        args.timestep_limit = env_spec.timestep_limit
    cfg = args.__dict__
    np.random.seed(args.seed)

    # Setup environment and filters
    aa = env.__dict__['allowed_actions']
    action_mapping = {i: aa[i] for i in range(len(aa))}
    of = ObFilter(20, 15)
    af = ActFilter(action_mapping)
    envf = FilteredEnv(env, ob_filter=of, act_filter=af, skiprate=(3,7))

    print envf.observation_space, envf.action_space

    agent = agent_ctor(envf.observation_space, envf.action_space, cfg)
    COUNTER = 0

    if args.use_hdf:
        if args.load_snapshot:
            hdf = load_h5_file(args)
            key = hdf["agent_snapshots"].keys()[-1]
            latest_snapshot = hdf["agent_snapshots"][key]
            agent = cPickle.loads(latest_snapshot.value)
            COUNTER = int(key)
        else:
            hdf = prepare_h5_file(args)

    gym.logger.setLevel(logging.WARN)

    print COUNTER

    def callback(stats):
        global COUNTER
        COUNTER += 1
        # Print stats
        print "*********** Iteration %i ****************" % COUNTER
        print tabulate(filter(lambda (k, v): np.asarray(v).size == 1,
                              stats.items()))  #pylint: disable=W0110
        # Store to hdf5
        if args.use_hdf:
            if args.snapshot_every and ((COUNTER % args.snapshot_every == 0) or
                                        (COUNTER == args.n_iter)):
                hdf['/agent_snapshots/%0.4i' % COUNTER] = np.array(cPickle.dumps(agent, -1))
        # Plot
        if args.plot:
            animate_rollout(envf, agent, min(2000, args.timestep_limit))

    run_policy_gradient_algorithm(envf, agent, callback=callback, usercfg=cfg)

    if args.use_hdf:
        hdf['env_id'] = env_spec.id
        try:
            hdf['env'] = np.array(cPickle.dumps(envf, -1))
        except Exception:
            print "failed to pickle env"  #pylint: disable=W0703

    env.monitor.close()
