from collections import OrderedDict
import os
import copy
from isaacgymenvs.utils.reformat import omegaconf_to_dict, print_dict
from isaacgymenvs.utils.rlgames_utils import RLGPUEnv, RLGPUAlgoObserver, get_rlgames_env_creator
from isaacgym import gymapi, gymtorch

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.distributions
from torch.optim.lr_scheduler import StepLR

import robot_learning
from robot_learning.algorithms.base_agent import BaseAgent
from robot_learning.algorithms.gail_agent import GAILAgent
from robot_learning.algorithms.ppo_agent import PPOAgent
from robot_learning.algorithms.expert_dataset import ExpertDataset
from robot_learning.networks.discriminator import Discriminator
from robot_learning.utils.normalizer import Normalizer
from robot_learning.utils.info_dict import Info
from robot_learning.utils.logger import logger
from robot_learning.utils.mpi import mpi_average
from robot_learning.utils.gym_env import value_to_space
from robot_learning.utils.pytorch import (
    get_ckpt_path,
    optimizer_cuda,
    count_parameters,
    sync_networks,
    sync_grads,
    to_tensor,
    obs2tensor,
)

from hydra.experimental import compose, initialize
from rl_games.common import env_configurations, vecenv
from rl_games.algos_torch.players import PpoPlayerContinuous
from rl_games.algos_torch.a2c_continuous import A2CAgent
import datetime
from isaacgymenvs.utils.utils import set_np_formatting, set_seed
from hydra.utils import to_absolute_path
import random
from rl_games.common import tr_helpers
import isaacgymenvs
import time
import gym
from gym.spaces import Box

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
def load_env_player():
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
    def _restore(agent, args):
        if 'checkpoint' in args and args['checkpoint'] is not None and args['checkpoint'] != '':
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
        player = A2CAgent(base_name='run', params=params)
        _restore(player, args)
        _override_sigma(player, args)
        player.init_tensors()

        player.mean_rewards = player.last_mean_rewards = -100500
        start_time = time.time()
        total_time = 0
        rep_count = 0
        # self.frame = 0  # loading from checkpoint

        return player

    #task_config in ["config_Pick.yaml", "config_Place.yaml","config_Screw.yaml"]
    task_configs = ["config_Pick.yaml", "config_Place.yaml", "config_Screw.yaml"]
    initialize(config_path="./method/robot_learning/cfg")  # change together with code in isaacgymenvs.make
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


    env_info = env_configurations.get_env_info(env)

    env_info_place = {'observation_space': Box(-float('inf'), float('inf'), [27]), 'action_space': Box(-1, 1, [12]),
                      'agents': 1,
                      'value_size': 1}
    env_info_screw = {'observation_space': Box(-float('inf'), float('inf'), [32]), 'action_space': Box(-1, 1, [12]),
                      'agents': 1,
                      'value_size': 1}

    env_infos = [env_info, env_info_place, env_info_screw]
    for index, confg_file in enumerate(task_configs):
        task = (confg_file.split(".")[0]).split("_")[1]
        task = "FactoryTaskNutBolt" + task
        tasks.append(task)

        cfg = compose(confg_file)
        cfg = completeconfig(cfg)
        cfg_dict = omegaconf_to_dict(cfg.task)
        cfg_dict['env']['numEnvs'] = 1
        cfgs.append(cfg_dict)
        players.append(load_player(cfg, env_infos[index]))
        players[-1].cast_obs(torch.tensor([1]))
        # players[-1].max_steps = 1000
        players[-1].print_stats = False
        #players[-1].vec_env = env
    players[1].max_steps = 200
    players[2].max_steps = 2048

    return cfgs, tasks, env_infos, env, players

def train_pps_agent(player, rollout):
    batch_dict = player.dataset.values_dict
    old_values = batch_dict['old_values']
    old_logp_actions = batch_dict['old_logp_actions']
    advantages = batch_dict['advantages']
    mus = batch_dict['mu']
    sigmas = batch_dict['sigma']
    mus_ = []
    sigmas_ = []
    for re,a,obs, old_value, old_logp_action, advantage,cmu, csigma in zip(rollout['rew'],rollout['ac'],rollout['ob'], old_values, old_logp_actions,advantages,mus, sigmas):
        batch_dict['returns'] = re
        batch_dict['actions'] = a
        batch_dict['obs'] = obs['obs']

        batch_dict['old_values'] = old_value.unsqueeze(0)
        batch_dict['old_logp_actions'] = old_logp_action
        batch_dict['advantages'] = advantage
        batch_dict['mu'] = cmu
        batch_dict['sigma'] = csigma

        a_loss, c_loss, entropy, kl, last_lr, lr_mul, cmu, csigma, b_loss = player.train_actor_critic(batch_dict)

        mus_.append(cmu)
        sigmas_.append(csigma)
    batch_dict['old_values'] = old_values
    batch_dict['old_logp_actions'] = old_logp_actions
    batch_dict['advantages'] = advantages
    batch_dict['mu'] = mus_
    batch_dict['sigma'] = sigmas_
    player.dataset.values_dict = batch_dict
    return

class PolicySequencingAgent(BaseAgent):
    def __init__(self, config, ob_space, ac_space, env_ob_space, task= None):
        self.task__ = task
        if task == 'PPS':
            super().__init__(config, ob_space)


            cfgs, tasks, env_infos, env, players = load_env_player()
            self.cfgs = cfgs
            self.tasks = tasks
            self.env_infos = env_infos
            self.env = env
            self.players = players
            self._ob_space = ob_space
            self._ac_space = env_infos[-1]['action_space']
            ob_space = self._ob_space
            ac_space = self._ac_space
            env_ob_space = ob_space
            self._num_agents = 3
            self._rl_agents = []
            """
            or i in range(self._num_agents):
                config_i = copy.copy(config)
                config_i.demo_path = config.ps_demo_paths[i]
                if config.ps_rl_algo == "gail":
                    self._rl_agents.append(
                        GAILAgent(config_i, ob_space, ac_space, env_ob_space)
                    )
            """
            if config.ps_use_tstar:
                self._discriminators = []
                if config.ps_discriminator_loss_type == "gan":
                    self._discriminator_loss = nn.BCEWithLogitsLoss()
                elif config.ps_discriminator_loss_type == "lsgan":
                    self._discriminator_loss = nn.MSELoss()
                for i in range(self._num_agents):
                    self._discriminators.append(
                        Discriminator(
                            config,
                            ob_space,
                            mlp_dim=config.ps_discriminator_mlp_dim,
                            activation=config.ps_discriminator_activation,
                        )
                    )
                self._network_cuda(config.device)

            if config.is_train and config.ps_use_tstar:
                self._discriminator_optims = []
                self._discriminator_lr_schedulers = []
                for i in range(self._num_agents):
                    # build optimizers
                    self._discriminator_optims.append(
                        optim.Adam(
                            self._discriminators[i].parameters(),
                            lr=config.ps_discriminator_lr,
                        )
                    )

                    # build learning rate scheduler
                    self._discriminator_lr_schedulers.append(
                        StepLR(
                            self._discriminator_optims[i],
                            step_size=self._config.max_global_step
                                      // self._config.rollout_length,
                            gamma=0.5,
                        )
                    )
            self._dataset = []
            """
            for i in range(self._num_agents):
                config_i = copy.copy(config)
                config_i.demo_path = config.ps_demo_paths[i]

                self._dataset.append(ExpertDataset(
                    config_i.demo_path,
                    config_i.demo_subsample_interval,
                    ac_space,
                    use_low_level=config.demo_low_level,
                    sample_range_start=config.demo_sample_range_start,
                    sample_range_end=config.demo_sample_range_end,
                ))
            """
            # expert dataset
            self.initial_states = [[],[],[]]  # for environment init state
            self.terminal_obs = [[],[],[]]  # for environment init state distribution
            self.initial_obs = [[],[],[]] # for constraining terminal state
            state_shape = Box(-float('inf'),float('inf'),[32])
            self.initial_state_dists = []
            for i in range(self._num_agents):
                self.initial_state_dists.append(Normalizer(ob_space, eps=0)) # for constraining terminal state
            """"
            for i in range(self._num_agents):
                self.initial_states.append(self._dataset[i].initial_states)
                self.initial_obs.append(self._dataset[i].initial_obs)
                self.terminal_obs.append(self._dataset[i].terminal_obs)
                state_shape = value_to_space(self.initial_states[i][0])
                self.initial_state_dists.append(Normalizer(state_shape, eps=0))
                self.initial_state_dists[i].update(self.initial_states[i])
                self.initial_state_dists[i].recompute_stats()
            """
            self._log_creation()
        else:
            super().__init__(config, ob_space)

            self._ob_space = ob_space
            self._ac_space = ac_space

            self._num_agents = len(config.ps_ckpts)
            self._rl_agents = []
            for i in range(self._num_agents):
                config_i = copy.copy(config)
                config_i.demo_path = config.ps_demo_paths[i]
                if config.ps_rl_algo == "gail":
                    self._rl_agents.append(
                        GAILAgent(config_i, ob_space, ac_space, env_ob_space)
                    )
                elif config.ps_rl_algo == "ppo":
                    self._rl_agents.append(
                        PPOAgent(config_i, ob_space, ac_space, env_ob_space)
                    )
                    self._rl_agents[i]._dataset = ExpertDataset(
                        config_i.demo_path,
                        config.demo_subsample_interval,
                        ac_space,
                        use_low_level=config.demo_low_level,
                        sample_range_start=config.demo_sample_range_start,
                        sample_range_end=config.demo_sample_range_end,
                    )

            for i, ckpt in enumerate(config.ps_ckpts):
                ckpt_path, ckpt_num = get_ckpt_path(ckpt, ckpt_num=None)
                assert ckpt_path is not None, "Cannot find checkpoint at %s" % ckpt_dir

                logger.warn("Load checkpoint %s", ckpt_path)
                self._rl_agents[i].load_state_dict(
                    torch.load(ckpt_path, map_location=self._config.device)["agent"]
                )

            if config.ps_use_tstar:
                self._discriminators = []
                if config.ps_discriminator_loss_type == "gan":
                    self._discriminator_loss = nn.BCEWithLogitsLoss()
                elif config.ps_discriminator_loss_type == "lsgan":
                    self._discriminator_loss = nn.MSELoss()
                for i in range(self._num_agents):
                    self._discriminators.append(
                        Discriminator(
                            config,
                            ob_space,
                            mlp_dim=config.ps_discriminator_mlp_dim,
                            activation=config.ps_discriminator_activation,
                        )
                    )
                self._network_cuda(config.device)

            if config.is_train and config.ps_use_tstar:
                self._discriminator_optims = []
                self._discriminator_lr_schedulers = []
                for i in range(self._num_agents):
                    # build optimizers
                    self._discriminator_optims.append(
                        optim.Adam(
                            self._discriminators[i].parameters(),
                            lr=config.ps_discriminator_lr,
                        )
                    )

                    # build learning rate scheduler
                    self._discriminator_lr_schedulers.append(
                        StepLR(
                            self._discriminator_optims[i],
                            step_size=self._config.max_global_step
                            // self._config.rollout_length,
                            gamma=0.5,
                        )
                    )

            # expert dataset
            self.initial_states = []  # for environment init state
            self.initial_state_dists = []  # for environment init state distribution
            self.initial_obs = []  # for constraining terminal state
            self.terminal_obs = []  # for constraining terminal state
            for i in range(self._num_agents):
                self.initial_states.append(self._rl_agents[i]._dataset.initial_states)
                self.initial_obs.append(self._rl_agents[i]._dataset.initial_obs)
                self.terminal_obs.append(self._rl_agents[i]._dataset.terminal_obs)
                state_shape = value_to_space(self.initial_states[i][0])
                self.initial_state_dists.append(Normalizer(state_shape, eps=0))
                self.initial_state_dists[i].update(self.initial_states[i])
                self.initial_state_dists[i].recompute_stats()

            self._log_creation()

    def __getitem__(self, key):
        if self.task__ == 'PPS':
            return self.players[key]
        else:
            return self._rl_agents[key]

    def _log_creation(self):
        if self._config.is_chef:
            logger.info("Creating a policy sequencing agent")

    def is_off_policy(self):
        return False

    def store_episode(self, rollouts, agent_idx):
        self._rl_agents[agent_idx].store_episode(rollouts)

    def state_dict(self):
        if self.task__ == 'PPS':
            ret = {
                "rl_agents": [agent.model.state_dict() for agent in self.players],
                "initial_states": self.initial_states,
                "initial_obs": self.initial_obs,
                "terminal_obs": self.terminal_obs,
                "initial_state_dists": [
                    dist.state_dict() for dist in self.initial_state_dists
                ],
            }
        else:
            ret = {
                "rl_agents": [agent.state_dict() for agent in self._rl_agents],
                "initial_states": self.initial_states,
                "initial_obs": self.initial_obs,
                "terminal_obs": self.terminal_obs,
                "initial_state_dists": [
                    dist.state_dict() for dist in self.initial_state_dists
                ],
            }
        if self._config.ps_use_tstar:
            ret["discriminators_state_dict"] = [
                d.state_dict() for d in self._discriminators
            ]
            ret["discriminator_optims_state_dict"] = [
                o.state_dict() for o in self._discriminator_optims
            ]

        return ret

    def load_state_dict(self, ckpt):
        for i in range(self._num_agents):
            self._rl_agents[i].load_state_dict(ckpt["rl_agents"][i])
            if self._config.ps_use_tstar:
                self._discriminators[i].load_state_dict(
                    ckpt["discriminators_state_dict"][i]
                )
                if self._config.is_train:
                    self._discriminator_optims[i].load_state_dict(
                        ckpt["discriminator_optims_state_dict"][i]
                    )
                    optimizer_cuda(self._discriminator_optims[i], self._config.device)
            if self._config.is_train:
                self.initial_state_dists[i].load_state_dict(
                    ckpt["initial_state_dists"][i]
                )
        if self._config.is_train:
            self.initial_states = ckpt["initial_states"]
            self.initial_obs = ckpt["initial_obs"]
            self.terminal_obs = ckpt["terminal_obs"]
        self._network_cuda(self._config.device)

    def _network_cuda(self, device):
        if self._config.ps_use_tstar:
            for i in range(self._num_agents):
                self._discriminators[i].to(device)

    def sync_networks(self):
        if self.task__ == "PPS":
            for i in range(self._num_agents):
                robot_learning.utils.pytorch.sync_networks(self.players[i].model)
                if self._config.ps_use_tstar:
                    sync_networks(self._discriminators[i])
        else:
            for i in range(self._num_agents):
                self._rl_agents[i].sync_networks()
                if self._config.ps_use_tstar:
                    sync_networks(self._discriminators[i])

    def update_normalizer(self, obs=None, i=None):
        if obs is not None and i is not None:
            self._rl_agents[i].update_normalizer(obs)

    def _predict_tstar_reward(self, ob, agent_idx):
        d = self._discriminators[agent_idx]
        d.eval()
        with torch.no_grad():
            ret = d(ob)
            eps = 1e-10
            s    = torch.sigmoid(ret)
            if self._config.ps_tstar_reward_type == "vanilla":
                reward = -(1 - s + eps).log()
            elif self._config.ps_tstar_reward_type == "gan":
                reward = (s + eps).log() - (1 - s + eps).log()
            elif self._config.ps_tstar_reward_type == "d":
                reward = ret
            elif self._config.ps_tstar_reward_type == "amp":
                ret = torch.clamp(ret, 0, 1) - 1
                reward = 1 - ret ** 2
        d.train()
        return reward

    def predict_tstar_reward(self, ob, agent_idx):
        #ob = self.normalize(ob)
        ob = to_tensor(ob, self._config.device)
        reward = self._predict_tstar_reward(ob, agent_idx)
        return reward.to('cuda:0')

    def train(self, agent_idx, roolout=None):
        train_info = Info()
        self._config.batch_size = 1
        if self._config.ps_use_tstar and agent_idx > 0:
            num_batches = (
                self._config.rollout_length
                // self._config.batch_size
                // self._config.ps_discriminator_update_freq
            )
            assert num_batches > 0

            self._discriminator_lr_schedulers[agent_idx].step()

            expert_dataset = self.initial_obs[agent_idx]
            policy_dataset = self.terminal_obs[agent_idx - 1]
            for _ in range(num_batches):
                # policy_data = self._rl_agents[agent_idx]._buffer.sample(
                #     self._config.batch_size
                # )
                idxs = np.random.randint(
                    0, len(policy_dataset), self._config.batch_size
                )
                states = [policy_dataset[idx] for idx in idxs]

                if isinstance(states[0], dict):
                    sub_keys = states[0].keys()
                    new_states = {
                        sub_key: np.stack([v[sub_key].cpu().numpy() for v in states])
                        for sub_key in sub_keys
                    }
                else:
                    new_states = np.stack(states)
                policy_data = {"ob": new_states}

                idxs = np.random.randint(
                    0, len(expert_dataset), self._config.batch_size
                )
                states = [expert_dataset[idx] for idx in idxs]
                if isinstance(states[0], dict):
                    sub_keys = states[0].keys()
                    new_states = {
                        sub_key: np.stack([v[sub_key].cpu().numpy() for v in states])
                        for sub_key in sub_keys
                    }
                else:
                    new_states = np.stack(states)
                expert_data = {"ob": new_states}

                _train_info = self._update_discriminator(
                    agent_idx, policy_data, expert_data
                )
                train_info.add(_train_info)
        if self.task__ == 'PPS':
            _train_info = train_pps_agent(self.players[agent_idx], roolout)
        else:
            _train_info = self._rl_agents[agent_idx].train()
        train_info.add(_train_info)

        # ob normalization?

        return train_info.get_dict(only_scalar=True)

    def _update_discriminator(self, i, policy_data, expert_data):
        info = Info()

        _to_tensor = lambda x: to_tensor(x, self._config.device)
        # pre-process observations
        p_o = policy_data["ob"]
        #p_o = self.normalize(p_o)
        p_o = _to_tensor(p_o)

        e_o = expert_data["ob"]
        #e_o = self.normalize(e_o)
        e_o = _to_tensor(e_o)

        p_logit = self._discriminators[i](p_o)
        e_logit = self._discriminators[i](e_o)

        if self._config.ps_discriminator_loss_type == "lsgan":
            p_output = p_logit
            e_output = e_logit
        else:
            p_output = torch.sigmoid(p_logit)
            e_output = torch.sigmoid(e_logit)

        p_loss = self._discriminator_loss(
            p_logit, torch.zeros_like(p_logit).to(self._config.device)
        )
        e_loss = self._discriminator_loss(
            e_logit, torch.ones_like(e_logit).to(self._config.device)
        )
        if p_logit.shape[1] > e_logit.shape[1]:
            e_logit_ = torch.zeros_like(p_logit)
            e_logit_[:,:e_logit.shape[1],:] = e_logit
            e_logit = e_logit_
        else:
            p_logit_ = torch.zeros_like(e_logit)
            p_logit_[:, :p_logit.shape[1], :] = p_logit
            p_logit = p_logit_
        logits = torch.cat([p_logit, e_logit], dim=0)
        entropy = torch.distributions.Bernoulli(logits=logits).entropy().mean()
        entropy_loss = -self._config.ps_entropy_loss_coeff * entropy

        #grad_pen = self._compute_grad_pen(i, p_o, e_o)
        #grad_pen_loss = self._config.ps_grad_penalty_coeff * grad_pen

        ps_loss = p_loss + e_loss + entropy_loss# + grad_pen_loss

        # update the discriminator
        self._discriminators[i].zero_grad()
        ps_loss.backward()
        sync_grads(self._discriminators[i])
        self._discriminator_optims[i].step()

        info["ps_disc_policy_output"] = p_output.mean().detach().cpu().item()
        info["ps_disc_expert_output"] = e_output.mean().detach().cpu().item()
        info["ps_disc_entropy"] = entropy.detach().cpu().item()
        info["ps_disc_policy_loss"] = p_loss.detach().cpu().item()
        info["ps_disc_expert_loss"] = e_loss.detach().cpu().item()
        info["ps_disc_entropy_loss"] = entropy_loss.detach().cpu().item()
        #info["ps_disc_grad_pen"] = grad_pen.detach().cpu().item()
        #info["ps_disc_grad_loss"] = grad_pen_loss.detach().cpu().item()
        info["ps_disc_loss"] = ps_loss.detach().cpu().item()

        return mpi_average(info.get_dict(only_scalar=True))

    def _compute_grad_pen(self, i, policy_ob, expert_ob):
        batch_size = self._config.batch_size
        alpha = torch.rand(batch_size, 1, device=self._config.device)

        def blend_dict(a, b, alpha):
            if isinstance(a, dict):
                return OrderedDict(
                    [(k, blend_dict(a[k], b[k], alpha)) for k in a.keys()]
                )
            elif isinstance(a, list):
                return [blend_dict(a[i], b[i], alpha) for i in range(len(a))]
            else:
                expanded_alpha = alpha.expand_as(a)
                ret = expanded_alpha * a + (1 - expanded_alpha) * b
                ret.requires_grad = True
                return ret

        interpolated_ob = blend_dict(policy_ob, expert_ob, alpha)
        inputs = list(interpolated_ob.values())

        interpolated_logit = self._discriminators[i](interpolated_ob)
        ones = torch.ones(interpolated_logit.size(), device=self._config.device)

        grad = autograd.grad(
            outputs=interpolated_logit,
            inputs=inputs,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        grad_pen = (grad.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen
