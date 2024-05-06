# Requirement

## Build your environment

Please modify the environment of Highwaydriving and record it!

## Exercise (30 minutes): Double DQN

In DQN, the same network is responsible for selecting and estimating the best next action (in the TD-target) and that may lead to over-estimation (the action which q-value is over-estimated will be chosen more often and this slow down training).

To reduce over-estimation, double q-learning (and then double DQN) was proposed. It decouples the action selection from the value estimation.

Concretely, in DQN, the target q-value is defined as:

$$Y^{DQN}_{t} = r_{t+1} + \gamma{Q}\left(s_{t+1}, \arg\max_{a}Q\left(s_{t+1}, a; \mathbb{\theta}_{target}\right); \mathbb{\theta}_{target}\right)$$

where the target network `q_net_target` with parameters $\mathbb{\theta}_{target}$ is used for both action selection and estimation, and can therefore be rewritten:

$$Y^{DQN}_{t} = r_{t+1} + \gamma \max_{a}{Q}\left(s_{t+1}, a; \mathbb{\theta}_{target}\right)$$

Double DQN uses the online network `q_net` with parameters $\mathbb{\theta}_{online}$ to select the action and the target network `q_net_target` to estimate the associated q-values:

$$Y^{DoubleDQN}_{t} = r_{t+1} + \gamma{Q}\left(s_{t+1}, \arg\max_{a}Q\left(s_{t+1}, a; \mathbb{\theta}_{online}\right); \mathbb{\theta}_{target}\right)$$

The goal in this exercise is for you to write the update method for `DoubleDQN`.

You will need to:

1. Sample replay buffer data using `self.replay_buffer.sample(batch_size)`

2. Compute the Double DQN target q-value using the next observations `replay_data.next_observation`, the online network `self.q_net`, the target network `self.q_net_target`, the rewards `replay_data.rewards` and the termination signals `replay_data.dones`. Be careful with the shape of each object ;)

3. Compute the current q-value estimates using the online network `self.q_net`, the current observations `replay_data.observations` and the buffer actions `replay_data.actions`

4. Compute the loss to train the q-network using L2 or Huber loss (`F.smooth_l1_loss`)


Link: https://paperswithcode.com/method/double-q-learning

Paper: https://arxiv.org/abs/1509.06461



# Env Install

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple gym
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple matplotlib
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple gym[toy_text]
```

**requirement.txt**

```
traci
sumolib
stable_baselines3
torch
shimmy
tensorboard
```

# DQN

The result store in  `log/dqn_highway_tensorboard/`

![image-20240501232828511](http://cdn.elapsedf.cn/img/image-20240501232828511.png)



# Simple HighwayDriving

## Env info

sumo_cfg: `3_lane_freeway.net.xml`

```xml
<net version="1.16" junctionCornerDetail="5" limitTurnSpeed="5.50" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/net_file.xsd">

    <location netOffset="0.00,0.00" convBoundary="0.00,0.00,2200.00,0.00" origBoundary="10000000000.00,10000000000.00,-10000000000.00,-10000000000.00" projParameter="!"/>

    <edge id="E5" from="J0" to="J2" priority="-1">
        <lane id="E5_0" index="0" speed="20" length="2200.00" shape="0.00,-8.00 2200.00,-8.00"/>
        <lane id="E5_1" index="1" speed="20" length="2200.00" shape="0.00,-4.80 2200.00,-4.80"/>
        <lane id="E5_2" index="2" speed="20" length="2200.00" shape="0.00,-1.60 2200.00,-1.60"/>
    </edge>

    <junction id="J0" type="dead_end" x="0.00" y="0.00" incLanes="" intLanes="" shape="0.00,0.00 0.00,-9.60"/>
    <junction id="J2" type="dead_end" x="2200.00" y="0.00" incLanes="E5_0 E5_1 E5_2" intLanes="" shape="2200.00,-9.60 2200.00,0.00"/>

</net>
```

Vehicle Num：2

Reward: 

```
self.w_speed = 1
self.w_p_time = 0.2
self.w_p_crash = 100
```



# HighwayDriving

We add 5 vehicles in the `3_lane_freeway.net.xml` and add one Car Type to make the environment more complex! 

## Some error recorded

NOTE: Please attention to add the `CarType` in the `freeway.rou.xml` when you add the `CarType`

<img src="http://cdn.elapsedf.cn/img/image-20240505180539396.png" alt="image-20240505180539396" style="zoom:67%;" />

When you meet `TypeError: cannot unpack non-iterable NoneType object`

You are probably need to write the `reset` function instead of super it only.

# DDQN Training

<img src="http://cdn.elapsedf.cn/img/image-20240505190754615.png" alt="image-20240505190754615" style="zoom:67%;" />

Check the Final result!

```
Tensorboard –-logdir=log
```

When you meet this error: `tensorboard: error: invalid choice: '–-logdir=dqn_highway_tensorboard' (choose from 'serve', 'dev')`

you could change the **Relative Path** to the **Absolute Path**



## Result

**Note that the result store in  `dqn_highway_tensorboard/`**

![image-20240505191714543](http://cdn.elapsedf.cn/img/image-20240505191714543.png)

![image-20240505201235626](http://cdn.elapsedf.cn/img/image-20240505201235626.png)
