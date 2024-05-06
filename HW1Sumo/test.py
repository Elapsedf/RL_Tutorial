# 基于SUMO搭建简单的超车环境
"""
此python文件基于SUMO搭建简单的超车环境
"""
import gymnasium as gym
import numpy as np

from gymnasium import spaces
from gymnasium.spaces.box import Box

import sumolib
import traci




class SimpleHighwayDriving(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode, label=None) -> None:
        super().__init__()

        # sumo
        self.label = label
        self.sumo_config = "RL_Tutorial\envs\cfg\\freeway.sumo.cfg" #RL_Tutorial\envs\cfg\freeway.sumo.cfg
        self.arguments = ["--lanechange.duration", "0.85", "--quit-on-end"]
        add_args = ["--delay", "100", "-c"] # 设置仿真延迟，范围0~1000
        self.arguments.extend(add_args)
        self.sumo_cmd = [sumolib.checkBinary('sumo')]
        self.arguments.append(self.sumo_config)
        self.already_running = False

        # env
        self._target_location = 2100
        self.init_speed = 5
        self.single_step = 1  # 1step for 1s simulation
        self.lane_counts = 1
        self.control_id = 'controled_0'

        # reward 
        self.w_speed = 1
        self.w_p_time = 0.2
        self.w_p_crash = 100

        #左转、右转、车道保持
        self.n_actions = 3
        self.action_space = spaces.Discrete(self.n_actions)

        # 左前、左后、右前、右后、正前方车辆
        self.surrounding_num = 5  
        F = 3  # diff_speed, diff_x_pos, diff_y_pos
        self.observation_space = Box(low=-np.inf,
                                     high=np.inf,
                                     shape=(self.surrounding_num * F, ),
                                     dtype=np.float64)
        
        assert render_mode is None or render_mode in self.metadata[
            "render_modes"]
        self.render_mode = render_mode

    def add_vehicles(self):

        # RL 控制车辆
        traci.vehicle.add(self.control_id,
                          "route",
                          departPos=0,
                          departSpeed=self.init_speed,
                          departLane=0,
                          typeID='CarB')
        traci.vehicle.setLaneChangeMode(self.control_id, 0)

        # 其他车辆
        traci.vehicle.add('veh_0',
                          "route",
                          departPos=20,
                          departSpeed=self.init_speed,
                          departLane=0,
                          typeID='CarA')
        traci.vehicle.setLaneChangeMode('veh_0', 0)   # 禁止车辆的换道
        traci.vehicle.setSpeedMode('veh_0', 0) # 禁用的车辆加减速

    def _get_obs(self):
        surrounding_vehs = []
        current_state = []
        speed_ego = traci.vehicle.getSpeed(self.control_id)
        x_ego, y_ego = traci.vehicle.getPosition(self.control_id)

        modes = [
            0b000,
            0b001,
            0b011,
            0b010,
        ]  #左前、左右、右前、右后车辆
        for mode in modes:
            veh = traci.vehicle.getNeighbors(self.control_id, mode=mode)
            if veh != ():
                surrounding_vehs.append(veh[0][0])
            else:
                surrounding_vehs.append('')
        header = traci.vehicle.getLeader(self.control_id)
        if not header is None:  # 前车
            surrounding_vehs.append(header[0])
        else:
            surrounding_vehs.append('')
        for veh in surrounding_vehs:
            if veh == '':
                x_diff = 0
                y_diff = 0
                speed_diff = 0
            else:
                speed = traci.vehicle.getSpeed(veh)
                x, y = traci.vehicle.getPosition(veh)
                speed_diff = abs(speed - speed_ego)
                x_diff = abs(x - x_ego)
                y_diff = abs(y - y_ego)
            current_state.append(x_diff)
            current_state.append(y_diff)
            current_state.append(speed_diff)
        return np.array(current_state)

    def _get_info(self, **kwargs):
        crash = True if len(kwargs['crash_ids']) > 0 else False
        return {
            "simulation step": self.count,
            "crash": crash,
        }

    def reset(self, seed=None, options=None):

        if not self.already_running:

            # 是否以可视化方式启动sumo
            if self.render_mode == "human":
                print("Creating a sumo-gui.")
                self.sumo_cmd = [sumolib.checkBinary('sumo-gui')] 
            else:
                print("No gui will display.")
            self.sumo_cmd.extend(self.arguments)

            traci.start(self.sumo_cmd)
            self.already_running = True
        else:
            traci.load(self.arguments)

        self.count = 0

        self.add_vehicles()

        self.count += self.single_step
        traci.simulationStep(self.count) # 仿真进行到count秒

        if self.render_mode == "human":
            traci.gui.trackVehicle("View #0", self.control_id)
            traci.gui.setZoom("View #0", 1000)

        observation = self._get_obs()
        info = {}

        return observation, info

    # 将强化学习模型输出作用于控制车辆
    def _apply_rl_action(self, action):
        ego_lane = traci.vehicle.getLaneIndex(self.control_id)
        if action == 1:
            target_lane = min(self.lane_counts, ego_lane + 1)
        elif action == 2:
            target_lane = max(0, ego_lane - 1)
        else:
            target_lane = ego_lane

        traci.vehicle.changeLane(self.control_id, target_lane, duration=0) # 立刻换道

    def _is_done(self):
        # 碰撞或者超车完成

        terminated = False
        truncated = False
        
        crash_ids = traci.simulation.getCollidingVehiclesIDList()

        pos = traci.vehicle.getPosition(self.control_id)[0]

        if pos >= self._target_location:
            terminated = True
            # print("{0} success!".format(self.control_id))
        if self.control_id in crash_ids:
            terminated = True
            # print('crashing!!! ')

        return terminated,truncated, crash_ids

    def _get_reward(self, **kwargs):

        # 速度奖励、碰撞惩罚、时间惩罚
        unit = 1

        speed_reward = traci.vehicle.getSpeed(self.control_id)

        time = traci.vehicle.getDeparture(self.control_id)
        time_penalty = np.array(traci.simulation.getTime() - time)

        total_crash_penalty = len(kwargs['crash_ids']) * unit

        reward = self.w_speed * speed_reward - self.w_p_time * time_penalty - self.w_p_crash * total_crash_penalty
        return np.array(reward)

    # 推进一次仿真步
    def step(self, action):

        self._apply_rl_action(action)
        self.count += self.single_step
        traci.simulationStep(self.count)

        terminated,truncated, crash_ids = self._is_done()
        reward = self._get_reward(crash_ids=crash_ids)
        observation = self._get_obs()
        info = self._get_info(crash_ids=crash_ids)

        return observation, reward, terminated, truncated, info

    def close(self):
        traci.close()

# 增加环境的复杂度

# class HighwayDriving(SimpleHighwayDriving):
#     def __init__(self, render_mode, label=None) -> None:
#         super().__init__(render_mode, label)
#         # put your code here
    
#     def reset(self, seed=None, options=None):
#         super().reset(seed, options)
#         # put your code here
#         pass
    
#     def add_vehicles(self):
#         pass
import random
class HighwayDriving(SimpleHighwayDriving):
    def __init__(self, render_mode, label=None, num_vehicles=5, vehicle_types=None) -> None:
        super().__init__(render_mode, label)
        self.num_vehicles = num_vehicles
        self.vehicle_types = vehicle_types if vehicle_types is not None else ['CarA', 'CarB', 'CarC']

    def reset(self, seed=None, options=None):
        super().reset(seed, options)
        if not self.already_running:

            # 是否以可视化方式启动sumo
            if self.render_mode == "human":
                print("Creating a sumo-gui.")
                self.sumo_cmd = [sumolib.checkBinary('sumo-gui')] 
            else:
                print("No gui will display.")
            self.sumo_cmd.extend(self.arguments)

            traci.start(self.sumo_cmd)
            self.already_running = True
        else:
            traci.load(self.arguments)

        self.count = 0

        self.add_vehicles()

        self.count += self.single_step
        traci.simulationStep(self.count) # 仿真进行到count秒

        if self.render_mode == "human":
            traci.gui.trackVehicle("View #0", self.control_id)
            traci.gui.setZoom("View #0", 1000)

        observation = self._get_obs()
        info = {}

        return observation, info
        # self.add_vehicles()

    def add_vehicles(self):
        # RL 控制车辆
        traci.vehicle.add(self.control_id,
                          "route",
                          departPos=0,
                          departSpeed=self.init_speed,
                          departLane=0,
                          typeID='CarB')
        traci.vehicle.setLaneChangeMode(self.control_id, 0)
        for i in range(self.num_vehicles):
            veh_id = f'veh_{i}'
            vehicle_type = random.choice(self.vehicle_types)
            depart_pos = random.uniform(10, 50)  # 随机选择车辆出发位置
            depart_speed = random.uniform(5, 15)  # 随机选择车辆出发速度
            depart_lane = random.randint(0, self.lane_counts - 1)  # 随机选择车辆出发车道
            traci.vehicle.add(veh_id,
                              "route",
                              departPos=depart_pos,
                              departSpeed=depart_speed,
                              departLane=depart_lane,
                              typeID=vehicle_type)
            traci.vehicle.setLaneChangeMode(veh_id, 0)  # 禁止车辆的换道
            traci.vehicle.setSpeedMode(veh_id, 0)  # 禁用的车辆加减速

from torch.nn import functional as F
import torch as th
import pdb
from stable_baselines3 import DQN
class DoubleDQN(DQN):
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            ### YOUR CODE HERE
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            # Do not backpropagate gradient to the target network
            with th.no_grad():
                # Compute the next Q-values using the target network
                next_q_values = self.q_net_target(replay_data.next_observations)
                # Decouple action selection from value estimation
                # Compute q-values for the next observation using the online q net
                next_q_values_online = self.q_net(replay_data.next_observations)
                # pdb.set_trace()
                # Select action with online network
                next_actions_online = next_q_values_online.max(dim=1).indices
                # Estimate the q-values for the selected actions using target q network
                row_indices = np.arange(len(next_actions_online))
                next_q_values = next_q_values[row_indices, next_actions_online]
                # Avoid potential broadcast issue
                next_q_values = next_q_values.reshape(-1, 1)
                # 1-step TD target
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates
            current_q_values = self.q_net(replay_data.observations)

            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())

            # Check the shape
            assert current_q_values.shape == target_q_values.shape

            # Compute loss (L2 or Huber loss)
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            # losses.append(loss.item())

            ### END OF YOUR CODE
            
            losses.append(loss.item())

            # Optimize the q-network
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("RL_Tutorial\\train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("RL_Tutorial\\train/loss", np.mean(losses))
    
from stable_baselines3.common.evaluation import evaluate_policy




# env = HighwayDriving(render_mode='human')
env = HighwayDriving(render_mode=None)

model = DoubleDQN("MlpPolicy", env, verbose=1, tensorboard_log="./dqn_highway_tensorboard/")

mean_reward, std_reward = evaluate_policy(model,
                                          env,
                                          n_eval_episodes=5,
                                          deterministic=False)
print('\n')
print(mean_reward)
print(std_reward)
print('\n')
print('-----------------------------------')

model.learn(total_timesteps=1e4)

mean_reward, std_reward = evaluate_policy(model,
                                          env,
                                          n_eval_episodes=5,
                                          deterministic=True)
print('-----------------------------------')
print('\n')
print(mean_reward)
print(std_reward)
print('\n')
env.close()