import numpy as np
import logging

from .envs.fabric_velocity_control_env \
    import FabricVelocityControlEnv

class FabricDomainBuilder():
        
    def build_from_config(domain, initial, action, org_action, num_picker,task, 
                          disp=None, seed=None, horizon=None, 
                          save_control_step_info=False):
        if num_picker > 2:
            raise ValueError("num_picker should be less than 2")

        ## Put the default config here
        config = {
            "headless": False,
            "random_seed": 0,
            "use_cached_states": True,
            "save_cached_states": False,
            "recolour_config": False,
            "num_picker": 2,
            'save_control_step_info': save_control_step_info,
            'save_image_dim': (256, 256),
            'control_horizon': 2000,
            'context': {},
            "picker_low": [[-0.62, 0.02, -0.62], [-0.62, 0.02, -0.62]],
            "picker_high": [[0.62, 1.0, 0.62], [0.62, 1.0, 0.62]],
            "picker_initial_pos": [[0.2, 0.2, 0.2], [-0.2, 0.2, 0.2]],
            'action_horizon': 30,
        }


        
        config.update(FabricDomainBuilder.return_config_from_domain(domain))
        config.update(FabricDomainBuilder.return_config_from_initial_state(domain, initial))
        config.update(FabricDomainBuilder.return_config_from_action(action, num_picker))
        config.update(FabricDomainBuilder.return_config_from_task(task, initial, domain))
        config['action_horizon'] = int(horizon) if horizon is not None else config['action_horizon']

        if disp is not None:
            disp = True if disp == 'True' else False
            config['headless'] = not disp

        if seed is not None:
            config['random_seed'] = int(seed)
        
        #print('config keys', config)
        
         #### Define base env
        from .picker_action_wrappers.pixel_pick_and_place_wrapper \
                  import PixelPickAndPlaceWrapper
        env = FabricVelocityControlEnv(config)
       
        
        ### Put Action Wrappers
        if action == 'pixel-pick-and-place':

            kwargs = {
                'motion_trajectory': 'rectangular',
                'pick_height': 0.028,
                'place_height':  0.05,
                'pick_lower_bound': [-1, -1],
                'pick_upper_bound': [1, 1],
                'place_lower_bound': [-1, -1],
                'place_upper_bound': [1, 1],
                # 'intermidiate_height_ratio': 0.4,
                # "minimum_intermidiate_height": 0.07,
                # "maximum_intermidiate_height": 0.1,
                # "release_height": 0.05,
                "prepare_height": 0.05,
                'fix_pick_height': True,
                'fix_place_height': True,
                'velocity': 0.05,
                'action_dim': num_picker
            }
            if domain in ['real2sim-towels', 'real2sim-square-fabric']:
                #kwargs['pick_height'] = 0.03
                kwargs['motion_trajectory'] = 'rectangular'
                kwargs['prepare_height'] = 0.05
            kwargs['action_horizon'] = config['action_horizon']
            if num_picker <=2:
                pass
            else:
                raise NotImplementedError
            env = PixelPickAndPlaceWrapper(env, **kwargs)

        elif action == 'pixel-pick-and-place-z':

            kwargs = {
                'motion_trajectory': 'triangle_with_height_ratio',
                'pick_height': 0.028,
                'place_height':  0.06,
                'pick_lower_bound': [-1, -1, 0],
                'pick_upper_bound': [1, 1, 1.5],
                'place_lower_bound': [-1, -1, 0],
                'place_upper_bound': [1, 1, 1.5],
                'intermidiate_height_ratio': 0.4,
                "minimum_intermidiate_height": 0.07,
                "maximum_intermidiate_height": 0.15,
                "release_height": 0.09,
                'fix_pick_height': False,
                'fix_place_height': False,
                "prepare_height": 0.2,
                'velocity': 0.05,
                'action_dim': num_picker
            }
            kwargs['action_horizon'] = config['action_horizon']
            if num_picker <= 2:
                pass
            else:
                raise NotImplementedError
            
            env = PixelPickAndPlaceWrapper(env, **kwargs)
        
        elif action == 'world-pick-and-place':
            kwargs = {
                'motion_trajectory': 'rectangular',
                "prepare_height": 0.1,
                'velocity': 0.05,
                'action_dim': num_picker
            }
            kwargs['action_horizon'] = config['action_horizon']
            
            from .picker_action_wrappers.world_pick_and_place_wrapper \
                import WorldPickAndPlaceWrapper
            env = WorldPickAndPlaceWrapper(env, **kwargs)

        elif action == 'velocity-grasp':
            if num_picker == 1:
                pass
            else:
                raise NotImplementedError
            
            from .picker_action_wrappers.velocity_grasp \
                import VelocityGrasp
            env = VelocityGrasp(env, action_repeat=10, 
                                max_interactive_step=1000)
        else:
            ## Raise with error message
            print("Action {} is not supported".format(action))
            raise NotImplementedError
        
        
        
        #print('here here')
        ### Put Task Wrapper
        if task == 'flattening':
            from .task_wrappers.towels.flattening_wrapper \
                import FlatteningWrapper
            env = FlatteningWrapper(env, domain=domain, initial=initial)
        elif task == 'diagonal-folding':
            #print('hello')
            from .task_wrappers.towels.diagonal_folding_wrapper \
                import DiagonalFoldingWrapper
            env = DiagonalFoldingWrapper(env, domain=domain, initial=initial, action=org_action)
        elif task == 'corners-edge-inward-folding':
            from .task_wrappers.towels.corners_edge_inward_folding_wrapper \
                import CornersEdgeInwardFoldingWrapper
            env = CornersEdgeInwardFoldingWrapper(env, domain=domain, initial=initial, action=org_action)

        elif task == 'diagonal-cross-folding':
            from .task_wrappers.towels.diagonal_cross_folding_wrapper \
                import DiagonalCrossFoldingWrapper
            env = DiagonalCrossFoldingWrapper(env, domain=domain, initial=initial, action=org_action)
               

        elif task == 'one-corner-inward-folding':
            from ...softgym.task_wrappers.rect_fabric.one_corner_inward_folding_wrapper \
                import OneCornerInwardFoldingWrapper
            env = OneCornerInwardFoldingWrapper(env, domain=domain, initial=initial, action=org_action)
        elif task == 'canonical-one-corner-inward-folding':
            from ...softgym.task_wrappers.rect_fabric.one_corner_inward_folding_wrapper \
                import OneCornerInwardFoldingWrapper
            env = OneCornerInwardFoldingWrapper(env, canonical=True, domain=domain, initial=initial)
        
        elif task == 'double-corner-inward-folding':
            from ...softgym.task_wrappers.rect_fabric.double_corner_inward_folding_wrapper \
                import DoubleCornerInwardFoldingWrapper
            env = DoubleCornerInwardFoldingWrapper(env, domain=domain, initial=initial, action=org_action)
        elif task == 'canonical-double-corner-inward-folding':
            from ...softgym.task_wrappers.rect_fabric.double_corner_inward_folding_wrapper \
                import DoubleCornerInwardFoldingWrapper
            env = DoubleCornerInwardFoldingWrapper(env, canonical=True, domain=domain, initial=initial)
        
        elif task == 'all-corner-inward-folding':
            from ...softgym.task_wrappers.rect_fabric.all_corner_inward_folding_wrapper \
                import AllCornerInwardFoldingWrapper
            env = AllCornerInwardFoldingWrapper(env, domain=domain, initial=initial, action=org_action)
        elif task == 'canonical-all-corner-inward-folding':
            from ...softgym.task_wrappers.rect_fabric.all_corner_inward_folding_wrapper \
                import AllCornerInwardFoldingWrapper
            env = AllCornerInwardFoldingWrapper(env, canonical=True, domain=domain, initial=initial)
        
        
        ## Following tasks are for both square and rectangular fabrics.
        elif task == 'rectangular-folding':
            from ...softgym.task_wrappers.rect_fabric.rectangular_folding_wrapper \
                import RectangularFoldingWrapper
            env = RectangularFoldingWrapper(env, domain=domain, initial=initial, action=org_action)

        elif task == 'canonical-rectangular-folding':
            from ...softgym.task_wrappers.rect_fabric.rectangular_folding_wrapper \
                import RectangularFoldingWrapper
            env = RectangularFoldingWrapper(env, canonical=True, domain=domain, initial=initial)

        elif task == 'side-folding':
            from ...softgym.task_wrappers.rect_fabric.side_folding_wrapper \
                import SideFoldingWrapper
            env = SideFoldingWrapper(env, domain=domain, initial=initial, action=org_action)
        elif task == 'canonical-side-folding':
            from ...softgym.task_wrappers.rect_fabric.side_folding_wrapper \
                import SideFoldingWrapper
            env = SideFoldingWrapper(env, canonical=True, domain=domain, initial=initial)

        elif task == 'double-side-folding':
            from ...softgym.task_wrappers.rect_fabric.double_side_folding_wrapper \
                import DoubleSideFoldingWrapper
            env = DoubleSideFoldingWrapper(env, domain=domain, initial=initial, action=org_action)
        elif task == 'canonical-double-side-folding':
            from ...softgym.task_wrappers.rect_fabric.double_side_folding_wrapper \
                import DoubleSideFoldingWrapper
            env = DoubleSideFoldingWrapper(env, canonical=True, domain=domain, initial=initial)

        elif task == 'double-side-folding':
            from ...softgym.task_wrappers.rect_fabric.double_side_folding_wrapper \
                import DoubleSideFoldingWrapper
            env = DoubleSideFoldingWrapper(env, domain=domain, initial=initial)
        elif task == 'canonical-double-side-folding':
            from ...softgym.task_wrappers.rect_fabric.double_side_folding_wrapper \
                import DoubleSideFoldingWrapper
            env = DoubleSideFoldingWrapper(env, canonical=True, domain=domain, initial=initial)
       

        else:
            logging.error("[softgym builder] task {} is not supported".format(task))
            raise NotImplementedError
        
        
        return env
        
    
    def return_config_from_domain(domain):
        config = {}

        if domain == 'mono-square-fabric':
            config['num_variations'] = 1000
            config['cached_states_path'] = "mono_square_fabric.pkl"
            config['eval_tiers'] = {
                4: [0, 10, 15, 20, 22, 25, 26, 27, 30, 31, 34, 35, 38, 39, 41, 42, 53, 56, 59, 60, 72, 75, 81, 87, 88, 90, 94],
                
                3: [1, 3, 16, 32, 45, 50, 54, 55, 57, 64, 65, 67, 71, 74, 79, 83, 86, 89, 91, 97, 98, 99],
                
                2: [5, 11, 21, 23, 28, 46, 52, 69, 70, 92],
                
                1: [36, 66, 96],
                
                0: [33, 49, 76]
                
            }
            config['video_episodes'] = [0, 10, 1, 3, 5, 11, 23, 36, 33]
            config['val_episodes'] = [2, 4, 6]
        
        
        elif domain == 'rainbow-rect-fabric':
            config['num_variations'] = 5000
            config['cached_states_path'] = "rainbow_rect_fabric.pkl"
            config['eval_tiers'] = {
                
                
                7: [6, 109, 197, 243, 320, 438, 465] ,
                
                6: [1, 22, 28, 30, 44, 49, 50] ,
                
                5: [0, 136, 159, 166, 187, 193, 208],

                4: [21, 68, 150, 189, 198, 249, 307] ,
                
                3: [8, 206, 212, 277, 323, 360, 474],
                
                2: [37, 55, 65, 69, 104, 112, 140],
                
                1: [26, 40, 43, 54, 56, 64, 75],
                
                0: [177, 216, 286, 329, 375, 443, 446]
                
            }
            config['val_episodes'] = [6, 109, 197]
            # first two eps of each tier
            config['video_episodes'] = [8, 206, 37, 55, 177, 216, 6, 109, 1, 22, 0, 136, 21, 68]
        
        elif domain == 'rainbow-square-fabric':
            config['num_variations'] = 2000
            config['cached_states_path'] = "rainbow_square_fabric.pkl"
            config['eval_tiers'] = {
                

                3: [10, 15, 28, 29, 36, 37, 46],
                
                2:  [33, 38, 41, 47, 49, 60, 66],
                
                1:   [4, 6, 8, 13, 18, 32, 44],
                
                0: [9, 11, 22, 31, 40, 55, 58]
                
            }

            config['video_episodes'] = [i for i in range(10)]
            config['val_episodes'] = [i for i in range(3)]
        
        
        elif domain in ['real2sim-towels', 'real2sim-square-fabric']:
            config['num_variations'] = 1000
            config['cached_states_path'] = "{}.pkl".format(domain)
            config['eval_tiers'] = { 0: [i for i in range(30)] }
            config['video_episodes'] = [i for i in range(10)]
            config['val_episodes'] = [i+30 for i in range(3)]
        
        elif domain == "ffn-rect-fabric":
            config['num_variations'] = 30
            config['cached_states_path'] = "vcd_square_fabric.pkl"
            config['eval_tiers'] = { 0: [i for i in range(30)] }
            config['video_episodes'] = [i for i in range(10)]
            config['val_episodes'] = [i for i in range(3)]
        elif domain == "ffn-square-fabric":
            config['num_variations'] = 30

            config['eval_tiers'] = { 0: [i for i in range(30)] }
            config['video_episodes'] = [i for i in range(10)]
            config['val_episodes'] = [i for i in range(3)]
        
        elif domain == "ffnm-rect-fabric":
            config['num_variations'] = 30

            config['eval_tiers'] = { 0: [i for i in range(30)] }
            config['video_episodes'] = [i for i in range(10)]
            config['val_episodes'] = [i for i in range(3)]
        
        elif domain == "ffnm-square-fabric":
            config['num_variations'] = 30
            config['eval_tiers'] = { 0: [i for i in range(30)] }
            config['video_episodes'] = [i for i in range(10)]
            config['val_episodes'] = [i for i in range(3)]
        
        elif domain == "ffmr-square-fabric":
            config['num_variations'] = 40
            config['eval_tiers'] = { 0: [i for i in range(30)] }
            config['video_episodes'] = [i for i in range(10)]
            config['val_episodes'] = [i for i in range(3)]
        
        elif domain == "ffmr-rect-fabric":
            config['num_variations'] = 40
            config['eval_tiers'] = { 0: [i for i in range(30)] }
            config['video_episodes'] = [i for i in range(10)]
            config['val_episodes'] = [i for i in range(3)]
            
        else:
            print("Target object <{}> not supported".format(domain))
            raise NotImplementedError
        
        return config
    
    def return_config_from_initial_state(domain, initial_state):
        
        config = {'initial_state': initial_state, 'context': {}}

        if domain == 'rainbow-rect-fabric':
            config['context']['rectangular'] = True
            config['context']['size'] = {
                'width': {
                    'lower_bound': 0.2,
                    'upper_bound': 0.7
                },

                'length': {
                    'lower_bound': 0.2,
                    'upper_bound': 0.7
                }
            }

            config['context']['colour'] = {
                'front_colour': {
                    'lower_bound': [0.0, 0.0, 0.0],
                    'upper_bound': [1.0, 1.0, 1.0]
                },
                'back_colour': {
                    'lower_bound': [0.0, 0.0, 0.0],
                    'upper_bound': [1.0, 1.0, 1.0]
                },
            }
        
        if domain == 'rainbow-square-fabric': 
            config['context']['rectangular'] = False
            config['context']['size'] = {
                'width': {
                    'lower_bound': 0.2,
                    'upper_bound': 0.7
                },

                'length': {
                    'lower_bound': 0.2,
                    'upper_bound': 0.7
                }
            }

            config['context']['colour'] = {
                'front_colour': {
                    'lower_bound': [0.0, 0.0, 0.0],
                    'upper_bound': [1.0, 1.0, 1.0]
                },
                'back_colour': {
                    'lower_bound': [0.0, 0.0, 0.0],
                    'upper_bound': [1.0, 1.0, 1.0]
                },
            }
        #print('domain', domain)
        if domain in ['real2sim-towels', 'real2sim-square-fabric']:
            #print("HELELLELE")
            if 'square' in domain:
                config['context']['rectangular'] = False
            else:
                config['context']['rectangular'] = True
            config['context']['size'] = {
                'width': {
                    'lower_bound': 0.15,
                    'upper_bound': 0.45
                },

                'length': {
                    'lower_bound': 0.15,
                    'upper_bound': 0.45
                }
            }
            
            config['context']['colour'] = {
                'front_colour': {
                    'lower_bound': [0.0, 0.0, 0.0],
                    'upper_bound': [1.0, 1.0, 1.0]
                },
                'back_colour': {
                    'lower_bound': [0.0, 0.0, 0.0],
                    'upper_bound': [1.0, 1.0, 1.0]
                },
            }
                
            config['context']['colour_mode'] = 'both_same'
            config['grasp_mode'] = {
                'around': 0.9,
                'miss': 0.1
            }
            #print('here')


        if initial_state == 'crumple':
            config['use_cached_states'] = True
            config['save_cached_states'] = True
            config['context']['position'] = 0.6
            config['context']['rotation'] = True
            config['context']['state'] = True
            config['context']['flip_face'] = 0.5
            
            if domain == 'vcd-rect-fabric':
                config['context']['rectangular'] = True
                config['context']['size'] = {
                    'width': {
                        'lower_bound': 0.19,
                        'upper_bound': 0.21
                    },

                    'length': {
                        'lower_bound': 0.31,
                        'upper_bound': 0.34
                    }
                }
            
            if domain == 'vcd-square-fabric':
                config['context']['size'] = {
                    'width': {
                        'lower_bound': 0.25,
                        'upper_bound': 0.28
                    }
                }
                config['context']['rectangular'] = False

            


        elif initial_state == 'flatten':
            config['use_cached_states'] = False
            config['num_variations'] = 1000
            config['eval_tiers'] = { 0: [i for i in range(30)] }
            #config['eval_tiers'][0][24] = 31
            config['video_episodes'] = [i for i in range(10)]
            config['val_episodes'] = [30+i for i in range(3)]
            #print('context', config['context'])
            config['context'].update({
                
                'position': 0.6,
                'rotation': True,
                'state': False,
                'flip_face': 0.5,
                'all_visible': True
            })
        
        elif initial_state == 'random_flatten':
            config['use_cached_states'] = False
            config['num_variations'] = 1000
            config['eval_tiers'] = { 0: [i for i in range(30)] }
            config['video_episodes'] = [i for i in range(10)]
            config['val_episodes'] = [30+i for i in range(3)]
            
            config['context'].update({
                
                'position': 0.6,
                'rotation': True,
                'state': False,
                'flip_face': 0.5,
                'all_visible': False
            })
           

            

        elif initial_state == 'canonical':
            config['context'] = {
                
                'position': 0,
                'rotation': False,
                'state': False,
                'flip_face': 0.5
            }
            config['eval_tiers'] = {
                0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                    10, 11, 12, 13, 14, 15, 16,
                    17, 18, 19, 20, 21, 22, 23,
                    24, 25, 26, 27, 28, 29]
            }
            config['video_episodes'] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            config['use_cached_states'] = False
            config['num_variations'] = 100
        
        elif initial_state == 'centre-flatten':
            config['use_cached_states'] = False
            config['num_variations'] = 1000
            config['eval_tiers'] = { 0: [i for i in range(30)] }
            config['video_episodes'] = [i for i in range(10)]
            config['val_episodes'] = [i for i in range(3)]
            
            config['context'] = {
                
                'position': 0,
                'rotation': True,
                'state': False,
                'flip_face': 0
            }

            if domain == 'ffmr-rect-fabric':
                config['context']['rectangular'] = True
                config['context']['size'] = {
                    'length': {
                        'lower_bound': 0.32,
                        'upper_bound': 0.37
                    },

                    'width': {
                        'lower_bound': 0.22,
                        'upper_bound': 0.33
                    }
                }
            if domain == 'ffmr-square-fabric':
                config['context']['rectangular'] = False
                config['context']['size'] = {
                    

                    'width': {
                        'lower_bound': 0.32,
                        'upper_bound': 0.37
                    }
                }

        elif initial_state == 'crumple':
            pass
        else:
            raise NotImplementedError
    
        return config
    
    def return_config_from_action(action, num_picker):
        config = {}
        if action in ['pixel-pick-and-place', 'pixel-pick-and-place-z', 'world-pick-and-place']:
            config.update({
                "picker_radius": 0.015,
                "picker_initial_pos": [[0.55, 0.2, 0.55], [-0.55, 0.2, 0.55]],
                "action_dim": num_picker
            })
        elif action == 'velocity-grasp':
            config.update({
                "picker_radius": 0.015,
            })
        else:
            ### Raise Error telling that such action is invalid
            print("Action <{}> not supported".format(action))
            raise NotImplementedError
        
        return config
    
    def return_config_from_task(task, init, domain):
        config = {}
        if task in ['flattening', 'canonical-flattening']:
            #config['reward_mode'] = "hoque_ddpg"
            config['action_horizon'] = 30 # 30
            #pass
        elif task in ['one-step-folding']:
            config['action_horizon'] = 1
            
        elif 'folding' in task:
            #config['reward_mode'] = "normalised_particle_distance"
            config['action_horizon'] = 10
            if init == 'crumple':
                config['action_horizon'] += 30
            pass
        elif task == 'all':
            #config['reward_mode'] = "normalised_particle_distance"
            #config['action_horizon'] = 25
            pass
        else:
            logging.error("[softgym builder] task {} is not supported".format(task))
            raise NotImplementedError
        
        
        if domain in ['real2sim-towels', 'real2sim-square-fabric']:
            
            config['camera_params'] = {
                'default_camera':{
                    'pos': np.array([-0.0, 0.65, 0]),
                    'angle': np.array([0, -90 / 180. * np.pi, 0.]),
                    'width': 720,
                    'height': 720},
            }
            config['cloth_param'] = {
                'size': [0.3, 0.3],
                'pos': [-0.15, 0.0, 0.15],
                'stiff': [2.0, 0.5, 1.0]
            }
            config['mass'] = 0.0054
        
        
        return config