"""
Collects policy sequencing rollouts (PolicySequencingRolloutRunner class).
"""

import pickle
import os
import time

import numpy as np
from mujoco_py.builder import MujocoException

from robot_learning.algorithms.rollout import Rollout, RolloutRunner
from robot_learning.utils.logger import logger
from robot_learning.utils.info_dict import Info
from robot_learning.utils.gym_env import get_non_absorbing_state, zero_value
import torch
def env_set_subtask(env, subtask, agent):
    from policy_sequencing_agent import reset_env
    env.cfg = agent.cfgs[subtask]
    env.task = agent.tasks[subtask]
    reset_env(env)

def env_step_(player, env, actions):
    actions = player.preprocess_actions(actions)
    obs, rewards, dones, infos = env.step(actions)

    if player.is_tensor_obses:
        if player.value_size == 1:
            rewards = rewards.unsqueeze(1)
        return player.obs_to_tensors(obs), rewards.to(player.ppo_device), dones.to(player.ppo_device), infos
    else:
        if player.value_size == 1:
            rewards = np.expand_dims(rewards, axis=1)
        return player.obs_to_tensors(obs), torch.from_numpy(rewards).to(player.ppo_device).float(), torch.from_numpy(
            dones).to(player.ppo_device), infos
def expand_state(states):
    if isinstance(states, list):
        for state in states:
            if isinstance(state, dict):
                for i in state.keys():
                    state[i] = state[i].cpu().numpy()
                    temp = np.zeros((state[i].shape[0], 32))
                    temp[:state[i].shape[0], :state[i].shape[1]] = state[i]
                    state[i] = torch.from_numpy(temp).to('cuda:0')
            else:
                return states
        return states
    elif isinstance(states, dict):
        for i in states.keys():
            states[i] = states[i].cpu()
            temp = torch.zeros((states[i].shape[0], 32))
            temp[:states[i].shape[0], :states[i].shape[1]] = states[i]
            states[i] = temp.to("cuda:0")
        return states
    else:
        temp = torch.zeros((states.shape[0], 32))
        temp[:states.shape[0], :states.shape[1]] = states
    return temp
class PolicySequencingRolloutRunner(RolloutRunner):
    """
    Run rollout given environment and multiple sub-policies.
    """

    def __init__(self, config, env, env_eval, agent):
        """
        Args:
            config: configurations for the environment.
            env: training environment.
            env_eval: testing environment.
            agent: policy.
        """
        super().__init__(config, env, env_eval, agent)
        if agent.task__ == 'PPS':
            self._n_subtask = 3
        else:
            self._n_subtask = env.num_subtask()
        self._subtask = 0
        self._init_sampling = 0
        self._last_subtask = -1
        # initialize env with stored states
        if config.ps_load_init_states and config.is_train:
            for i, path in enumerate(config.ps_load_init_states):
                if path and i < self._n_subtask:
                    path = os.path.expanduser(path)
                    with open(path, "rb") as f:
                        states = pickle.load(f)

                        self._agent.initial_states[i].extend(expand_state(states))
                        #self._agent.initial_state_dists[i].update(temp)

    def _reset_env(self, env, subtask=None, num_connects=None, episode = 0):
        """ Resets the environment and return the initial observation. """
        if subtask is None:
            subtask = self._subtask
        if self._agent.task__ == 'PPS':
            env_set_subtask(env, subtask, self._agent)
        else:
            env.set_subtask(env, subtask)
        p = np.random.rand()
        init_qpos = None
        self._init_sampling = 0

        if subtask > 0 and len(self._agent.initial_states[subtask]) > 0:
            if p < self._config.ps_env_init_from_dist:
                #init_qpos = self._agent.initial_state_dists[subtask].sample(1)
                self._init_sampling = 1
            elif p > 1 - self._config.ps_env_init_from_states:
                #init_qpos = np.random.choice(self._agent.initial_states[subtask])
                self._init_sampling = 2
        if self._agent.task__ == 'PPS':
            # if reset env with init_qpos, realize here
            env.subtask = subtask
            if subtask==0:
                if episode == 0:
                    init_obs = env.reset()
                else:
                    env.reset()
                    env.reset_buf = torch.ones(
                        env.num_envs, device=env.device, dtype=torch.long)
                    init_obs = env.compute_observations()
                    init_obs = {'obs':init_obs}

            else:
                init_obs = env.compute_observations()
                init_obs = {'obs':init_obs}
            player = self._agent[subtask]
            player.obs = init_obs
            player.set_eval()
            with torch.no_grad():
                if player.is_rnn:
                    batch_dict = player.play_steps_rnn()
                else:
                    batch_dict = player.play_steps_v2()

            player.set_train()
            player.curr_frames = batch_dict.pop('played_frames')
            player.prepare_dataset(batch_dict)

            player.algo_observer.after_steps()
            if player.has_central_value:
                player.train_central_value()
            return init_obs
        else:
            env.set_init_qpos(init_qpos)
            try:
                ret = env.reset()
            except MujocoException:
                logger.error("Fail to initialize env with %s", init_qpos)
                env.set_init_qpos(None)
                ret = env.reset()

        return ret

    def switch_subtask(self, subtask):
        self._subtask = subtask

    def run(
        self,
        is_train=True,
        every_steps=None,
        every_episodes=None,
        log_prefix="",
        step=0,
    ):
        """
        Collects trajectories and yield every @every_steps/@every_episodes.

        Args:
            is_train: whether rollout is for training or evaluation.
            every_steps: if not None, returns rollouts @every_steps
            every_episodes: if not None, returns rollouts @every_epiosdes
            log_prefix: log as @log_prefix rollout: %s
        """
        if every_steps is None and every_episodes is None:
            raise ValueError("Both every_steps and every_episodes cannot be None")

        config = self._config
        env = self._env if is_train else self._env_eval
        agent = self._agent
        il = hasattr(agent[0], "predict_reward")

        # initialize rollout buffer

        rollout = Rollout()
        reward_info = Info()
        ep_info = Info()
        episode = 0

        if agent.task__ == 'PPS':
            done = False
            ep_len = 0
            ep_rew = 0
            ep_rew_rl = 0
            if il:
                ep_rew_il = 0

            ob_init = ob_next = self._reset_env(env, subtask= self._subtask, num_connects=1,episode = episode)
            subtask = self._subtask
            next_subtask = subtask + 1
            reward_tstar = 0

            # run rollout
            while not done:
                ob = ob_next

                # sample action from policy
                agent[subtask].obs = ob
                ac = agent[subtask].get_action_values(ob)['actions']
                ac_before_activation = ac


                ob_next, reward, done, info = env_step_(agent[subtask], env, ac)
                if subtask == 2:
                    successs = env._get_curr_successes()
                    done = torch.logical_or(successs, env._get_curr_failures(successs))
                done = done.all()

                if 'successes' in info:
                    init_l = []
                    init_next = []
                    obs_init = ob_init['obs']
                    obs_next = ob_next['obs']
                    if subtask<2:
                        obs_next = env.get_taks_observations(subtask+1)
                    for index_, x in enumerate(list(info['successes_'].cpu().numpy())):
                        if x:

                            init_l.append(obs_init[index_,:])
                            init_next.append(obs_next[index_,:])

                    agent.initial_obs[subtask].append(expand_state({'obs':torch.stack(init_l,dim=0)}))
                    agent.terminal_obs[subtask].append(expand_state({'obs':torch.stack(init_next,dim=0)}))
                    info["episode_success_state"] = expand_state({'obs':torch.stack(init_next,dim=0)})

                    ob_init = ob_next
                    # add termination state regularization reward
                    if next_subtask < self._n_subtask and config.ps_use_tstar:
                        if subtask<2:
                            reward_tstar = agent.predict_tstar_reward(expand_state(env.get_taks_observations(subtask+1)), next_subtask)
                            reward += config.ps_tstar_reward * reward_tstar
                if il:
                    reward_il = agent[subtask].predict_reward(ob, ob_next, ac)
                    reward_rl = (
                        (1 - config.gail_env_reward) * reward_il
                        + config.gail_env_reward * reward * config.reward_scale
                    )
                else:
                    reward_rl = reward * config.reward_scale

                step += 1
                ep_len += 1
                ep_rew += reward
                ep_rew_rl += reward_rl
                if il:
                    ep_rew_il += reward_il

                if done and ep_len < env.max_episode_length:
                    done_mask = 0  # -1 absorbing, 0 done, 1 not done
                else:
                    done_mask = 1

                rollout.add(
                    {
                        "ob": ob,
                        "ob_next": ob_next,
                        "ac": ac,
                        "ac_before_activation": ac_before_activation,
                        "done": done,
                        "rew": reward,
                        "done_mask": done_mask,  # -1 absorbing, 0 done, 1 not done
                    }
                )


                reward_info.add(info)

                if every_steps is not None and step % every_steps == 0:
                    yield rollout.get(), ep_info.get_dict(only_scalar=True)


            # add successful final states to the next subtask's initial states
            if (
                config.is_train
                and config.ps_use_terminal_states
                and "episode_success_state" in reward_info.keys()
                and (self._init_sampling > 0 or subtask == 0)
                and next_subtask < self._n_subtask
            ):
                state = reward_info["episode_success_state"]
                self._agent.initial_states[next_subtask].extend(state)
                #self._agent.initial_state_dists[next_subtask].update(state)

            # compute average/sum of information
            reward_info_dict = reward_info.get_dict(reduction="sum", only_scalar=True)
            reward_info_dict.update({"len": ep_len, "rew": ep_rew, "rew_rl": ep_rew_rl})
            if il:
                reward_info_dict["rew_il"] = ep_rew_il
            reward_info_dict["rew_tstar"] = reward_tstar
            ep_info.add(reward_info_dict)

            logger.info(
                log_prefix + " rollout: %s",
                {
                    k: v
                    for k, v in reward_info_dict.items()
                    if k not in self._exclude_rollout_log and np.isscalar(v)
                },
            )

            episode += 1
            if every_episodes is not None and episode % every_episodes == 0:
                yield rollout.get(), ep_info.get_dict(only_scalar=True)
        else:
            while True:
                done = False
                ep_len = 0
                ep_rew = 0
                ep_rew_rl = 0
                if il:
                    ep_rew_il = 0

                ob_init = ob_next = self._reset_env(env, subtask= self._subtask, num_connects=1,episode = episode)
                subtask = self._subtask
                next_subtask = subtask + 1
                reward_tstar = 0
                if agent.task__ == 'PPS':
                    is_determenistic = agent[subtask].is_determenistic
                # run rollout
                while not done:
                    ob = ob_next

                    # sample action from policy
                    if agent.task__ == 'PPS':
                        ac = agent[subtask].get_action(ob, is_determenistic)
                        ac_before_activation = ac

                        ob_next, reward, done, info = agent[subtask].env_step(env, ac)
                        if subtask == 2:
                            successs = agent[subtask]._get_curr_successes()
                            done = torch.logical_or(successs, agent[subtask]._get_curr_failures(successs))
                        done = done.all()
                        if 'successes' in info and info['successes']:
                            agent.initial_obs[subtask].append(expand_state(ob_init))
                            agent.terminal_obs[subtask].append(expand_state(ob_next))
                            info["episode_success_state"] = expand_state(ob_next)

                            ob_init = ob_next
                            # add termination state regularization reward
                            if next_subtask < self._n_subtask and config.ps_use_tstar:
                                reward_tstar = agent.predict_tstar_reward(expand_state(ob), next_subtask)
                                reward += config.ps_tstar_reward * reward_tstar

                    else:
                        ac, ac_before_activation = agent[subtask].act(ob, is_train=is_train)
                    # take a step
                        ob_next, reward, done, info = env.step(ac)

                        # if subtask succeeds
                        if "subtask" in info and subtask != info["subtask"]:
                            agent.initial_obs[subtask].append(ob_init)
                            agent.terminal_obs[subtask].append(ob_next)
                            ob_init = ob_next

                            # add termination state regularization reward
                            if next_subtask < self._n_subtask and config.ps_use_tstar:
                                reward_tstar = agent.predict_tstar_reward(ob, next_subtask)
                                reward += config.ps_tstar_reward * reward_tstar

                    if il:
                        reward_il = agent[subtask].predict_reward(ob, ob_next, ac)
                        reward_rl = (
                            (1 - config.gail_env_reward) * reward_il
                            + config.gail_env_reward * reward * config.reward_scale
                        )
                    else:
                        reward_rl = reward * config.reward_scale

                    step += 1
                    ep_len += 1
                    ep_rew += reward
                    ep_rew_rl += reward_rl
                    if il:
                        ep_rew_il += reward_il

                    if done and ep_len < env.max_episode_steps:
                        done_mask = 0  # -1 absorbing, 0 done, 1 not done
                    else:
                        done_mask = 1

                    rollout.add(
                        {
                            "ob": ob,
                            "ob_next": ob_next,
                            "ac": ac,
                            "ac_before_activation": ac_before_activation,
                            "done": done,
                            "rew": reward,
                            "done_mask": done_mask,  # -1 absorbing, 0 done, 1 not done
                        }
                    )

                    reward_info.add(info)

                    if config.absorbing_state and done_mask == 0:
                        absorbing_state = env.get_absorbing_state()
                        absorbing_action = zero_value(env.action_space)
                        rollout._history["ob_next"][-1] = absorbing_state
                        rollout.add(
                            {
                                "ob": absorbing_state,
                                "ob_next": absorbing_state,
                                "ac": absorbing_action,
                                "ac_before_activation": absorbing_action,
                                "rew": 0.0,
                                "done": 0,
                                "done_mask": -1,  # -1 absorbing, 0 done, 1 not done
                            }
                        )

                    if every_steps is not None and step % every_steps == 0:
                        yield rollout.get(), ep_info.get_dict(only_scalar=True)

                # add successful final states to the next subtask's initial states
                if (
                    config.is_train
                    and config.ps_use_terminal_states
                    and "episode_success_state" in reward_info.keys()
                    and (self._init_sampling > 0 or subtask == 0)
                    and next_subtask < self._n_subtask
                ):
                    state = reward_info["episode_success_state"]
                    self._agent.initial_states[next_subtask].extend(state)
                    #self._agent.initial_state_dists[next_subtask].update(state)

                # compute average/sum of information
                reward_info_dict = reward_info.get_dict(reduction="sum", only_scalar=True)
                reward_info_dict.update({"len": ep_len, "rew": ep_rew, "rew_rl": ep_rew_rl})
                if il:
                    reward_info_dict["rew_il"] = ep_rew_il
                reward_info_dict["rew_tstar"] = reward_tstar
                ep_info.add(reward_info_dict)

                logger.info(
                    log_prefix + " rollout: %s",
                    {
                        k: v
                        for k, v in reward_info_dict.items()
                        if k not in self._exclude_rollout_log and np.isscalar(v)
                    },
                )

                episode += 1
                if every_episodes is not None and episode % every_episodes == 0:
                    yield rollout.get(), ep_info.get_dict(only_scalar=True)

    def run_episode(self, is_train=True, record_video=False, partial=False):
        """
        Runs one episode and returns the rollout (mainly for evaluation).

        Args:
            is_train: whether rollout is for training or evaluation.
            record_video: record video of rollout if True.
            partial: run each subtask policy.
        """
        config = self._config
        env = self._env if is_train else self._env_eval
        agent = self._agent
        il = hasattr(agent[0], "predict_reward")

        # initialize rollout buffer
        rollout = Rollout()
        reward_info = Info()
        if partial:
            subtask = self._subtask
            num_connects = 1
        else:
            subtask = 0
            num_connects = None
            env.set_max_episode_steps(config.max_episode_steps * 2)

        done = False
        ep_len = 0
        ep_rew = 0
        ep_rew_rl = 0
        if il:
            ep_rew_il = 0
        reward_tstar = 0

        ob_next = self._reset_env(env, subtask, num_connects)

        record_frames = []
        if record_video:
            record_frames.append(self._store_frame(env, ep_len, ep_rew))

        # run rollout
        while not done:
            ob = ob_next

            # sample action from policy
            ac, ac_before_activation = agent[subtask].act(ob, is_train=is_train)

            # take a step
            ob_next, reward, done, info = env.step(ac)
            if il:
                reward_il = agent[subtask].predict_reward(ob, ob_next, ac)

            next_subtask = subtask + 1
            if "subtask" in info and subtask != info["subtask"]:
                subtask = info["subtask"]

                # replace reward
                if next_subtask < self._n_subtask and config.ps_use_tstar:
                    reward_tstar = agent.predict_tstar_reward(ob, next_subtask)
                    reward += config.ps_tstar_reward * reward_tstar

            if il:
                reward_rl = (
                    (1 - config.gail_env_reward) * reward_il
                    + config.gail_env_reward * reward * config.reward_scale
                )
            else:
                reward_rl = reward * config.reward_scale

            ep_len += 1
            ep_rew += reward
            ep_rew_rl += reward_rl
            if il:
                ep_rew_il += reward_il

            rollout.add(
                {
                    "ob": ob,
                    "ac": ac,
                    "ac_before_activation": ac_before_activation,
                    "done": done,
                    "rew": reward,
                }
            )

            reward_info.add(info)
            if record_video:
                frame_info = info.copy()
                if il:
                    frame_info.update(
                        {
                            "ep_rew_il": ep_rew_il,
                            "rew_il": reward_il,
                            "rew_rl": reward_rl,
                            "rew_tstar": reward_tstar,
                        }
                    )
                record_frames.append(self._store_frame(env, ep_len, ep_rew, frame_info))

        # add last observation
        rollout.add({"ob": ob_next})

        # compute average/sum of information
        ep_info = {"len": ep_len, "rew": ep_rew, "rew_rl": ep_rew_rl}
        if il:
            ep_info["rew_il"] = ep_rew_il
        ep_info["rew_tstar"] = reward_tstar
        if "episode_success_state" in reward_info.keys():
            ep_info["episode_success_state"] = reward_info["episode_success_state"]
        ep_info.update(reward_info.get_dict(reduction="sum", only_scalar=True))

        logger.info(
            "rollout: %s",
            {
                k: v
                for k, v in ep_info.items()
                if k not in self._exclude_rollout_log and np.isscalar(v)
            },
        )

        if not partial:
            env.set_max_episode_steps(config.max_episode_steps)

        return rollout.get(), ep_info, record_frames
