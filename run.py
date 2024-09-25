import os
import pandas as pd
import argparse
import numpy as np
import time
import cv2

from benchmarks.visualisation_utils import save_video as sv
from benchmarks.visualisation_utils import save_numpy_as_gif as sg
from benchmarks.visualisation_utils import plot_pick_and_place_trajectory as pt


class MyLogger():
    def __init__(self, log_dir):
        self.log_dir = log_dir
        #super().__init__(log_dir)

    def __call__(self, episode_config, result, filename=None):

        eid, save_video =  episode_config['eid'], episode_config['save_video']
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        if filename is None:
            filename = 'manupilation'
        
        if not os.path.exists(os.path.join(self.log_dir, filename)):
            os.makedirs(os.path.join(self.log_dir, filename))

        df_dict = {
            'episode_id': [eid],
            #'return': [result['return']]
        }

        evaluations = result['evaluation']
        evaluation_keys = list(evaluations .keys())
        for k in evaluation_keys:
            df_dict['evaluation/'+ k] = [evaluations[k]]

        
        df = pd.DataFrame.from_dict(df_dict)
        performance_file = \
            os.path.join(self.log_dir, filename, 'performance.csv'.format(filename))
        written = os.path.exists(performance_file)

        
        df.to_csv(
            performance_file, 
            mode= ('a' if written else 'w'), 
            header= (False if written else True)
        )

        result['actions'] = np.stack(result['actions'])
        T = result['actions'].shape[0]
        N = result['actions'].shape[1]
        result['actions'] = result['actions'].reshape(T, N, 2, -1)[:, :, :, :2]
        rgbs = [info['observation']['rgb'] for info in result['informations']]
        pt(
            np.stack(rgbs), result['actions'].reshape(T, -1, 4), # TODO: this is envionrment specific
            title='Episode {}'.format(eid), 
            # rewards=result['rewards'], 
            save_png = True, save_path=os.path.join(self.log_dir, filename, 'performance_visualisation'), col=10)

        
        if save_video and 'frames' in result:    
            sv(result['frames'], 
                os.path.join(self.log_dir, filename, 'performance_visualisation'),
                'episode_{}'.format(eid))

        if save_video and 'frames' in result:    
            sg(
                result['frames'], 
                os.path.join(
                    self.log_dir, 
                    filename, 
                    'performance_visualisation',
                    'episode_{}'.format(eid)
                )
            )
    
    def check_exist(self, episode_config, filename=None):
        eid = episode_config['eid']

        if filename is None:
            filename = 'manupilation'
        
        performance_file = \
            os.path.join(self.log_dir, filename, 'performance.csv')
        #print('performance_file', performance_file)

        if not os.path.exists(performance_file):
            return False
        df = pd.read_csv(performance_file)
        if len(df) == 0:
            return False
        
        ## Check if there is an entry with the same tier and eid, and return True if there is
        return len(df[(df['episode_id'] == eid)]) > 0

def perform(arena, agent, mode='eval', episode_config=None,
    collect_frames=False,
    update_agent_from_arena=lambda ag, ar: None):

    if mode == 'eval':
        arena.set_eval()
    elif mode == 'train':
        arena.set_train()
    elif mode == 'val':
        arena.set_val()
    else:
        raise ValueError('mode must be either train, eval, or val')
    

    res = {}
    rgbs = []
    depths = []
    internal_states = []
    informations = []
    #rewards = []
    actions = []
    phases = []
    action_time = []
    res['evaluation'] = {}
    
    #arena.set_save_control_step_info(collect_frames)
    if episode_config['save_video']:
        frames = []
       
    eid = episode_config['eid']
    agent.reset()
    information = arena.reset(episode_config)
    

    information['done'] = False
    informations.append(information)
    agent.init(information)
    #internal_states.append(agent.get_state().copy())
    #information['reward'] = 0
    evals = arena.evaluate()

    for k, v in evals.items():
        res['evaluation'][k] = [v]
    
    update_agent_from_arena(agent, arena)

    #agent.init_state(information)

    if ('save_goal' in episode_config) and episode_config['save_goal']:
        res['goal'] = arena.get_goal()
        #print('goal keys', res['goal'].keys())
   
    while not information['done']:
        start_time = time.time()
        
        action = agent.act(information)
        #print('perform action', action)
        phase = agent.get_phase()
        phases.append(phase)
        internal_states.append(agent.get_state().copy())

        end_time = time.time()
        elapsed_time = (end_time - start_time)
        action_time.append(elapsed_time)
        information = arena.step(action)
        informations.append(information)

        
        if episode_config['save_video']:
            frame = np.asarray(arena.get_frames())
            ## resize frames to 256x256
            frame = np.stack([cv2.resize(f, (256, 256)) for f in frame])
            #print('frame shape', frame.shape)
            frames.append(frame[:, :, :, :3])
            arena.clear_frames()

        #print('action', action)
        actions.append(action)
       
        evals = arena.evaluate()
        
        if 'normalised_coverage' in evals:
            print('evals', evals['normalised_coverage'])
        

       
        agent.update(information, action)
        
        information['done'] = information['done'] or agent.success() or arena.success() or agent.terminate()
      
        for k, v in evals.items():
            res['evaluation'][k].append(v)

        print('\n\nStep: {}, Eval: {}'.format(len(actions), evals))

       
    res['actions'] = actions #np.stack(actions)
    res['action_durations'] = np.asarray(action_time)
    internal_states.append(agent.get_state().copy())
    res['phases'] = np.stack(phases)
    res['informations'] = informations
    if episode_config['save_video']:
        res['frames'] = np.concatenate(frames, axis=0)
    res['internal_states'] = internal_states
    return res


def run(policy, env, episode_config, logger):
    
    filename = 'manipulation'
    
    if logger.check_exist(episode_config, filename):
        return
     
    res = perform(env, policy, 'eval', episode_config=episode_config,
                collect_frames=episode_config['save_video'])
    
   
    logger(episode_config, res, filename)

def main():

    ### Argument Definition
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--domain', default='real2sim-towels')
    parser.add_argument('--task', default='flattening')
    parser.add_argument('--policy', default='real2sim-smoothing')
    parser.add_argument('--initial', default='crumple')
    parser.add_argument('--eid', default=1, type=int)
   
    ## make the --disp keywords a boolean when set it becomes true

    parser.add_argument('--disp', action='store_true')
    parser.add_argument('--save_video', action='store_true')
    parser.add_argument('--log_dir', default='.')
    
    args = parser.parse_args()
    

    save_dir = f"{args.log_dir}/results/{args.domain}-{args.task}-{args.initial}/{args.policy}/"

    ### Initialise arena
    env_name = 'softgym|domain:{},initial:{},action:pixel-pick-and-place(1),task:{},disp:{}'\
        .format(args.domain, args.initial, args.task, args.disp)
    from benchmarks.builder import Builder
    env = Builder.build(env_name)


    from benchmarks.oracles.policies import NAME2POLICY
    params = {
        'action_low': [-1, -1, -1, -1],
        'action_high': [1, 1, 1, 1],

    }
    policy = NAME2POLICY[args.policy]()
    
    run(policy, 
        env, 
        episode_config={
            'eid': args.eid, 
            'save_video': args.save_video,
        },
        logger=MyLogger(save_dir))
    


if __name__ == '__main__':
    main()