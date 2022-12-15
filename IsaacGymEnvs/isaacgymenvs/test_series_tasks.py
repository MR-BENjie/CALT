# train.py
# Script to train policie   s in Isaac Gym
#
# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import datetime

import numpy as np

import isaacgym
import random
import os
import time
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path
from hydra.core.hydra_config import HydraConfig
import gym
from gym.spaces import Box
from hydra.experimental import compose, initialize
from isaacgym import gymapi, gymtorch
from isaacgymenvs.utils.reformat import omegaconf_to_dict, print_dict

from isaacgymenvs.utils.utils import set_np_formatting, set_seed
from isaacgymenvs.utils.rlgames_utils import RLGPUEnv, RLGPUAlgoObserver, get_rlgames_env_creator
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner
from rl_games.algos_torch import model_builder
from rl_games.common import tr_helpers
from rl_games.algos_torch.players import PpoPlayerContinuous

from isaacgymenvs.learning import amp_continuous
from isaacgymenvs.learning import amp_players
from isaacgymenvs.learning import amp_models
from isaacgymenvs.learning import amp_network_builder
import isaacgymenvs

## OmegaConf & Hydra Config

# Resolvers used in hydra configs (see https://omegaconf.readthedocs.io/en/2.1_branch/usage.html#resolvers)
pick_cfg = None
place_cfg = None
screw_cfg = None

def _restore(agent, args):
    if 'checkpoint' in args and args['checkpoint'] is not None and args['checkpoint'] !='':
        agent.restore(args['checkpoint'])

def _override_sigma(agent, args):
    if 'sigma' in args and args['sigma'] is not None:
        net = agent.model.a2c_network
        if hasattr(net, 'sigma') and hasattr(net, 'fixed_sigma'):
            if net.fixed_sigma:
                with torch.no_grad():
                    net.sigma.fill_(float(args['sigma']))
            else:
                print('Print cannot set new sigma because fixed_sigma is False')

def play(player, env, init_obs):
    render = player.render_env
    is_determenistic = player.is_determenistic
    sum_rewards = 0
    sum_steps = 0
    sum_game_res = 0
    games_played = 0
    has_masks = False
    has_masks_func = getattr(env, "has_action_mask", None) is not None

    op_agent = getattr(env, "create_agent", None)
    if op_agent:
        agent_inited = True
        #print('setting agent weights for selfplay')
        # self.env.create_agent(self.env.config)
        # self.env.set_weights(range(8),self.get_weights())

    if has_masks_func:
        has_masks = env.has_action_mask()

    need_init_rnn = player.is_rnn


    #obses = self.env_reset(self.env)
    #obs = env.reset()
    #return self.obs_to_torch(obs)
    obses = player.obs_to_torch(init_obs)
    batch_size = 1
    batch_size = player.get_batch_size(obses, batch_size)

    if need_init_rnn:
        player.init_rnn()
        need_init_rnn = False

    cr = torch.zeros(batch_size, dtype=torch.float32)
    steps = torch.zeros(batch_size, dtype=torch.float32)

    print_game_res = False

    for n in range(player.max_steps):
        if has_masks:
            masks = env.get_action_mask()
            action = player.get_masked_action(
                obses, masks, is_determenistic)
        else:
            action = player.get_action(obses, is_determenistic)


        obses, r, done, info = player.env_step(env, action)
        cr += r
        steps += 1

        if render:
            env.render(mode='human')
            time.sleep(player.render_sleep)

        all_done_indices = done.nonzero(as_tuple=False)
        done_indices = all_done_indices[::player.num_agents]
        done_count = len(done_indices)
        games_played += done_count
        s = env.check_success()

        if s.all() :
            return obses, s.float()
        #if env.task == "FactoryTaskNutBoltScrew":
        #   print(env.nut_com_angvel)
            #print(obses[:4,:])
        if env.task == "FactoryTaskNutBoltScrew":
            fa = env._get_curr_failures(s)
            if torch.logical_or(s,fa).all():
                print("all failures")
                return obses, s.float()

        if done_count > 0:
            if env.task == "FactoryTaskNutBoltPick":
                return obses, s.float()
            if player.is_rnn:
                for x in player.states:
                    x[:, all_done_indices, :] = x[:,all_done_indices, :] * 0.0

            cur_rewards = cr[done_indices].sum().item()
            cur_steps = steps[done_indices].sum().item()

            cr = cr * (1.0 - done.float())
            steps = steps * (1.0 - done.float())
            sum_rewards += cur_rewards
            sum_steps += cur_steps

            game_res = 0.0
            if isinstance(info, dict):
                if 'battle_won' in info:
                    print_game_res = True
                    game_res = info.get('battle_won', 0.5)
                if 'scores' in info:
                    print_game_res = True
                    game_res = info.get('scores', 0.5)

            if player.print_stats:
                if print_game_res:
                    print('reward:', cur_rewards/done_count,
                          'steps:', cur_steps/done_count, 'w:', game_res)
                else:
                    print('reward:', cur_rewards/done_count,
                          'steps:', cur_steps/done_count)

            sum_game_res += game_res
            if batch_size//player.num_agents == 1:
                break

    return obses, s.float()

def completeconfig(cfg):

    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{cfg.wandb_name}_{time_str}"

    # ensure checkpoints can be specified as relative paths
    if cfg.checkpoint:
        cfg.checkpoint = to_absolute_path(cfg.checkpoint)

    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    # set numpy formatting for printing only
    set_np_formatting()

    rank = int(os.getenv("LOCAL_RANK", "0"))
    if cfg.multi_gpu:
        # torchrun --standalone --nnodes=1 --nproc_per_node=2 train.py
        cfg.sim_device = f'cuda:{rank}'
        cfg.rl_device = f'cuda:{rank}'

    # sets seed. if seed is -1 will pick a random one
    cfg.seed += rank
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic, rank=rank)

    # register the rl-games adapter to use inside the runner

    return cfg

def load_config(params):
    seed = params.get('seed', None)
    if seed is None:
        seed = int(time.time())
    if params["config"].get('multi_gpu', False):
        seed += int(os.getenv("LOCAL_RANK", "0"))
    print(f"self.seed = {seed}")

    if seed:

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

        # deal with environment specific seed if applicable
        if 'env_config' in params['config']:
            if not 'seed' in params['config']['env_config']:
                params['config']['env_config']['seed'] = seed
            else:
                if params["config"].get('multi_gpu', False):
                    params['config']['env_config']['seed'] += int(os.getenv("LOCAL_RANK", "0"))

    config = params['config']
    config['reward_shaper'] = tr_helpers.DefaultRewardsShaper(**config['reward_shaper'])
    if 'features' not in config:
        config['features'] = {}
    config['features']['observer'] = RLGPUAlgoObserver()
    return params

def load_player(cfg, env_info):
    args = {
        'train': not cfg.test,
        'play': cfg.test,
        'checkpoint': cfg.checkpoint,
        'sigma': None
    }
    rlg_config_dict = omegaconf_to_dict(cfg.train)
    rlg_config_dict['params']['config']['env_info'] = env_info

    params = load_config(rlg_config_dict["params"])
    player = PpoPlayerContinuous(params)
    _restore(player, args)
    _override_sigma(player, args)

    return player

def reset_env(env):
    env._get_task_yaml_params()
    env.acquire_base_tensors()
    env._acquire_env_tensors()
    env._acquire_task_tensors()

    env.parse_controller_spec()
    env.refresh_base_tensors()
    env.refresh_env_tensors()
    env._refresh_task_tensors()
    env.obs_buf = torch.zeros((env.num_envs, env.cfg_task.env.numObservations), device=env.device, dtype=torch.float)
    env_ids = range(env.num_envs)
    env.dof_vel[env_ids] = 0.0  # shape = (num_envs, num_dofs)
    env.ctrl_target_dof_pos[env_ids] = env.dof_pos[env_ids]

    multi_env_ids_int32 = env.franka_actor_ids_sim[env_ids].flatten()
    env.gym.set_dof_state_tensor_indexed(env.sim,
                                          gymtorch.unwrap_tensor(env.dof_state),
                                          gymtorch.unwrap_tensor(multi_env_ids_int32),
                                          len(multi_env_ids_int32))
    env.gym.set_actor_root_state_tensor_indexed(env.sim,
                                                 gymtorch.unwrap_tensor(env.root_state),
                                                 gymtorch.unwrap_tensor(env.nut_actor_ids_sim[env_ids]),
                                                 len(env.nut_actor_ids_sim[env_ids]))
    env.dof_vel[env_ids, :] = torch.zeros_like(env.dof_vel[env_ids])

    # Set DOF state
    multi_env_ids_int32 = env.franka_actor_ids_sim[env_ids].flatten()
    env.gym.set_dof_state_tensor_indexed(env.sim,
                                          gymtorch.unwrap_tensor(env.dof_state),
                                          gymtorch.unwrap_tensor(multi_env_ids_int32),
                                          len(multi_env_ids_int32))
    env.root_linvel[:, env.bolt_actor_id_env] = torch.zeros((env.num_envs, 3), device=env.device, dtype=torch.float)
    env.root_angvel[:, env.bolt_actor_id_env] = torch.zeros((env.num_envs, 3), device=env.device, dtype=torch.float)



    if env.cfg_task.sim.disable_gravity:
        env.disable_gravity()
    else:
        env.enable_gravity(gravity_mag=env.cfg_base.sim.gravity_mag)
    if env.viewer is not None:
        env._set_viewer_params()


    env.reset_buf = torch.ones(env.num_envs, device=env.device, dtype=torch.long)

def series_task(task_configs):
    initialize(config_path="./cfg") #change together with code in isaacgymenvs.make
    cfgs = []
    tasks = []
    players = []

    cfg = compose(task_configs[0])
    cfg = completeconfig(cfg)

    def create_env_thunk(**kwargs):
        envs = isaacgymenvs.make(
            cfg.seed,
            cfg.task_name,
            cfg.task.env.numEnvs,
            cfg.sim_device,
            cfg.rl_device,
            cfg.graphics_device_id,
            cfg.headless,
            cfg.multi_gpu,
            cfg.capture_video,
            cfg.force_render,
            cfg,
            **kwargs,
        )
        if cfg.capture_video:
            envs.is_vector_env = True
            envs = gym.wrappers.RecordVideo(
                envs,
                f"videos/{run_name}",
                step_trigger=lambda step: step % cfg.capture_video_freq == 0,
                video_length=cfg.capture_video_len,
            )
        return envs
    vecenv.register('RLGPU',
                    lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))
    env_configurations.register('rlgpu', {
        'vecenv_type': 'RLGPU',
        'env_creator': create_env_thunk,
    })


    env = env_configurations.configurations['rlgpu']['env_creator']()
    """
    env = isaacgymenvs.make(
        seed=42,
        task="FactoryTaskNutBoltPick",
        num_envs=64,
        sim_device="cuda:0",
        rl_device="cuda:0",)
    """

    env_info = env_configurations.get_env_info(env)

    env_info_place = {'observation_space': Box(-float('inf'),float('inf'),[27]), 'action_space': Box(-1,1,[12]), 'agents': 1,
     'value_size': 1}
    env_info_screw = {'observation_space': Box(-float('inf'),float('inf'),[32]),  'action_space': Box(-1,1,[12]), 'agents': 1,
     'value_size': 1}

    env_infos = [env_info, env_info_place, env_info_screw]
    for index, confg_file in enumerate(task_configs):
        task = (confg_file.split(".")[0]).split("_")[1]
        task = "FactoryTaskNutBolt" + task
        tasks.append(task)

        cfg = compose(confg_file)
        cfg = completeconfig(cfg)
        cfg_dict = omegaconf_to_dict(cfg.task)
        cfg_dict['env']['numEnvs'] = 128
        cfgs.append(cfg_dict)
        players.append(load_player(cfg, env_infos[index]))
        #players[-1].max_steps = 1000
        players[-1].print_stats = False
    players[1].max_steps = 200
    if len(players) == 3:
        players[2].max_steps = 4098
    game_num = 100
    task_num = len(cfgs)
    success_rate = []
    dons = []
    for n in range(game_num):
        for i in range(task_num):

            env.cfg = cfgs[i]
            env.task = tasks[i]
            reset_env(env)

            print("do "+tasks[i]+" : ",end="")
            if i==0:
                if n ==0:
                    init_obs = env.reset()
                else:
                    env.reset()
                    env.reset_buf = torch.ones(
                        env.num_envs, device=env.device, dtype=torch.long)
                    init_obs = env.compute_observations()
            elif i>0:
                init_obs = env.compute_observations()

            init_obs , done = play(players[i], env, init_obs)
            #if i==2:
                #print(env.nut_dist_to_target)
            print(torch.mean(done))
            dons.append((done))
        s = dons[-1]
        success_rate.append((torch.mean(s)).cpu().numpy())
        print(torch.mean(s))
    print("___________________________\nsuccess : ",end="")
    print(np.mean(success_rate))
if __name__ == "__main__":
    series_task(["config_Pick.yaml", "config_Place.yaml","config_Screw.yaml"])

