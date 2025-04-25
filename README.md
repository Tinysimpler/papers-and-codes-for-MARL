This is a collection of Multi-Agent Reinforcement Learning (MARL) papers with code. I have selected some relatively important papers with open source code and categorized them by time and method based on the efforts of [Chen, Hao, Multi-Agent Reinforcement Learning Papers with Code](https://github.com/TimeBreaker/MARL-papers-with-code), thanks to him.

Then I'll update this collection for deeper study.

- [经典论文](#经典论文)
  - [算法](#算法)
  - [环境](#环境)
- [其他论文](#其他论文)
- [综述](#综述)
    - [**Recent Reviews (Since 2019)**](#recent-reviews-since-2019)
    - [**Other Reviews (Before 2019)**](#other-reviews-before-2019)
- [环境](#环境-1)
- [多优化目标](#多优化目标)
  - [综述类](#综述类)
  - [环境](#环境-2)
- [信用分配](#信用分配)
  - [值分解](#值分解)
  - [其他方法](#其他方法)
  - [策略梯度](#策略梯度)
- [多任务](#多任务)
- [通信](#通信)
  - [带宽限制](#带宽限制)
  - [无带宽限制](#无带宽限制)
  - [未分](#未分)
- [涌现](#涌现)
- [对手建模](#对手建模)
- [博弈论](#博弈论)
- [分层](#分层)
- [角色](#角色)
- [大规模](#大规模)
- [即兴协作](#即兴协作)
- [进化算法](#进化算法)
  - [综述类](#综述类-1)
- [团队训练](#团队训练)
- [课程学习](#课程学习)
- [平均场](#平均场)
- [迁移学习](#迁移学习)
- [元学习](#元学习)
- [公平性](#公平性)
- [奖励搜索](#奖励搜索)
  - [稠密奖励搜索](#稠密奖励搜索)
  - [稀疏奖励搜索](#稀疏奖励搜索)
  - [未分](#未分-1)
- [稀疏奖励](#稀疏奖励)
- [图神经网络](#图神经网络)
- [基于模型的](#基于模型的)
- [神经架构搜索NAS](#神经架构搜索nas)
- [安全学习](#安全学习)
- [单智能体到多智能体](#单智能体到多智能体)
- [动作空间](#动作空间)
- [多样性](#多样性)
- [分布式训练分布式执行](#分布式训练分布式执行)
- [离线多智能体强化学习](#离线多智能体强化学习)
- [对抗](#对抗)
  - [单智能体](#单智能体)
  - [多智能体](#多智能体)
  - [对抗通信](#对抗通信)
  - [评估](#评估)
- [模仿学习](#模仿学习)
- [训练数据](#训练数据)
- [优化器](#优化器)
- [待分类](#待分类)
- [致谢](#致谢)

# 经典论文

## 算法

| **Category**         | **Paper**                                                    | **Code**                                           | **Accepted at** | **Year** |
| -------------------- | ------------------------------------------------------------ | -------------------------------------------------- | --------------- | -------- |
| Independent Learning | [IQL：Multi-Agent Reinforcement Learning: Independent vs. Cooperative Agents](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.84.3701&rep=rep1&type=pdf) | https://github.com/oxwhirl/pymarl                  | ICML            | 1993     |
| Value Decomposition  | [VDN：Value-Decomposition Networks For Cooperative Multi-Agent Learning](https://arxiv.org/pdf/1706.05296) | https://github.com/oxwhirl/pymarl                  | AAMAS           | 2017     |
| Value Decomposition  | [QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](http://proceedings.mlr.press/v80/rashid18a/rashid18a.pdf) | https://github.com/oxwhirl/pymarl                  | ICML            | 2018     |
| Value Decomposition  | [QTRAN: Learning to Factorize with Transformation for Cooperative Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1905.05408) | https://github.com/oxwhirl/pymarl                  | ICML            | 2019     |
| Policy Gradient      | [COMA：Counterfactual Multi-Agent Policy Gradients](https://arxiv.org/abs/1705.08926) | https://github.com/oxwhirl/pymarl                  | AAAI            | 2018     |
| Policy Gradient      | [MADDPG：Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/pdf/1706.02275.pdf">Multi-Agent) | https://github.com/openai/maddpg                   | NIPS            | 2017     |
| Communication        | [BiCNet：Multiagent Bidirectionally-Coordinated Nets: Emergence of Human-level Coordination in Learning to Play StarCraft Combat Games](https://arxiv.org/abs/1703.10069) | https://github.com/Coac/CommNet-BiCnet             |                 | 2017     |
| Communication        | [CommNet：Learning Multiagent Communication with Backpropagation](https://arxiv.org/abs/1605.07736) | https://github.com/facebookarchive/CommNet         | NIPS            | 2016     |
| Communication        | [IC3Net：Learning when to Communicate at Scale in Multiagent Cooperative and Competitive Tasks](https://arxiv.org/abs/1812.09755) | https://github.com/IC3Net/IC3Net                   |                 | 2018     |
| Communication        | [RIAL/RIDL：Learning to Communicate with Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1605.06676) | https://github.com/iassael/learning-to-communicate | NIPS            | 2016     |
| Exploration          | [MAVEN：Multi-Agent Variational Exploration](https://arxiv.org/pdf/1910.07483) | https://github.com/starry-sky6688/MARL-Algorithms  | NIPS            | 2019     |

## 环境

| **Environment** | **Paper**                                                    | **KeyWords**                                                 | **Code**                                                     | **Accepted at** | **Year** | **Others**                                                   |
| --------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | --------------- | -------- | ------------------------------------------------------------ |
| StarCraft       | [The StarCraft Multi-Agent Challenge](https://arxiv.org/pdf/1902.04043) |                                                              | https://github.com/oxwhirl/smac                              | NIPS            | 2019     |                                                              |
| StarCraft       | [SMACv2: A New Benchmark for Cooperative Multi-Agent Reinforcement Learning](https://openreview.net/pdf?id=pcBnes02t3u) |                                                              | https://github.com/oxwhirl/smacv2                            |                 | 2022     |                                                              |
| StarCraft       | [Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative Tasks](https://arxiv.org/pdf/2006.07869) |                                                              | https://github.com/uoe-agents/epymarl                        | NIPS            | 2021     |                                                              |
| Football        | [Google Research Football: A Novel Reinforcement Learning Environment](https://ojs.aaai.org/index.php/AAAI/article/view/5878/5734) |                                                              | https://github.com/google-research/football                  | AAAI            | 2020     |                                                              |
| PettingZoo      | [PettingZoo: Gym for Multi-Agent Reinforcement Learning](https://proceedings.neurips.cc/paper/2021/file/7ed2d3454c5eea71148b11d0c25104ff-Paper.pdf) |                                                              | https://github.com/Farama-Foundation/PettingZoo              | NIPS            | 2021     |                                                              |
| Melting Pot     | [Scalable Evaluation of Multi-Agent Reinforcement Learning with Melting Pot](http://proceedings.mlr.press/v139/leibo21a/leibo21a.pdf) |                                                              | https://github.com/deepmind/meltingpot                       | ICML            | 2021     |                                                              |
| MuJoCo          | [MuJoCo: A physics engine for model-based control](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.296.6848&rep=rep1&type=pdf) |                                                              | https://github.com/deepmind/mujoco                           | IROS            | 2012     |                                                              |
| MALib           | [MALib: A Parallel Framework for Population-based Multi-agent Reinforcement Learning](https://arxiv.org/pdf/2106.07551) | 基于种群的多智能体强化学习自动课程学习可扩展速度比RLlib快5倍比OpenSpiel至少快3倍 | https://github.com/sjtu-marl/malib                           |                 | 2021     |                                                              |
| MAgent          | [MAgent: A many-agent reinforcement learning platform for artificial collective intelligence](https://ojs.aaai.org/index.php/AAAI/article/download/11371/11230) |                                                              | https://github.com/Farama-Foundation/MAgent                  | AAAI            | 2018     |                                                              |
| Neural MMO      | [Neural MMO: A Massively Multiagent Game Environment for Training and Evaluating Intelligent Agents](https://arxiv.org/pdf/1903.00784) |                                                              | https://github.com/openai/neural-mmo                         |                 | 2019     |                                                              |
| MPE             | [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://proceedings.neurips.cc/paper/2017/file/68a9750337a418a86fe06c1991a1d64c-Paper.pdf) |                                                              | https://github.com/openai/multiagent-particle-envs           | NIPS            | 2017     |                                                              |
| Pommerman       | [Pommerman: A multi-agent playground](https://arxiv.org/pdf/1809.07124.pdfâ€‹arxiv.org) |                                                              | https://github.com/MultiAgentLearning/playground             |                 | 2018     |                                                              |
| HFO             | [Half Field Offense: An Environment for Multiagent Learning and Ad Hoc Teamwork](https://www.cse.iitb.ac.in/~shivaram/papers/hmsks_ala_2016.pdf) |                                                              | https://github.com/LARG/HFO                                  | AAMAS Workshop  | 2016     |                                                              |
|                 | Multiagent Coordination Simulator                            | 基于局部信息的控制多种多智能体协调技术群体行为钉扎控制动态包围任意闭合曲线跟踪牧羊控制对抗恶意智能体的控制Python | https://github.com/tjards/multi-agent_sim?tab=readme-ov-file |                 | 2024     | ![img](https://e8bw0pe0za.feishu.cn/space/api/box/stream/download/asynccode/?code=OGE4NmMxMzZkZjg2ODRhY2Q0YmNlYWJjYzIyYzZkMTBfaDZwZGIxMzNYYXFoUDZMcnF1Z3VvVkppVmJOdjVpWjhfVG9rZW46UGZ4QWJTRUZwb2Qxd0d4bmNiTmNwYXFibmpoXzE3NDU1NDU4MzA6MTc0NTU0OTQzMF9WNA) |
|                 |                                                              | A collection of reference environments for offline reinforcement learningPython | https://github.com/Farama-Foundation/D4RL                    |                 |          |                                                              |
|                 |                                                              | a collection of discrete grid-world environments to conduct research on Reinforcement Learning | https://github.com/Farama-Foundation/Minigrid                |                 |          | ![img](https://e8bw0pe0za.feishu.cn/space/api/box/stream/download/asynccode/?code=YTE3NGFkNzkzOTczNTE2NThmYmYwNWU1Yjg1NjI2MTFfSGtXd3NaZExxYmV6a0tDOEl4S01xY3FRR2xCTWJMRUtfVG9rZW46QlpJUGJaSVd5b2pqTDF4YmFROWMwdzQxbnpnXzE3NDU1NDU4MzA6MTc0NTU0OTQzMF9WNA) |

# 其他论文

| **Category**                     | **Paper**                                                    | **Code**                                                     | **Accepted at** | **Year** |
| -------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | --------------- | -------- |
| Graph Neural Network             | [Multi-Agent Game Abstraction via Graph Attention Neural Network](https://ojs.aaai.org/index.php/AAAI/article/view/6211/6067) | https://github.com/starry-sky6688/MARL-Algorithms            | AAAI            | 2020     |
| Curriculum Learning              | [From Few to More: Large-Scale Dynamic Multiagent Curriculum Learning](https://arxiv.org/abs/1909.02790) | https://github.com/starry-sky6688/MARL-Algorithms            | AAAI            | 2020     |
| Curriculum Learning              | [EPC：Evolutionary Population Curriculum for Scaling Multi-Agent Reinforcement Learning](https://arxiv.org/pdf/2003.10423) | https://github.com/qian18long/epciclr2020                    | ICLR            | 2020     |
| Curriculum Learning/Emergent     | [Emergent Tool Use From Multi-Agent Autocurricula](https://arxiv.org/pdf/1909.07528) | https://github.com/openai/multi-agent-emergence-environments | ICLR            | 2020     |
| Curriculum Learning              | [Cooperative Multi-agent Control using deep reinforcement learning](http://ala2017.it.nuigalway.ie/papers/ALA2017_Gupta.pdf) | https://github.com/sisl/MADRL                                | AAMAS           | 2017     |
| Role                             | [ROMA: Multi-Agent Reinforcement Learning with Emergent Roles](https://openreview.net/pdf?id=RQP2wq-dbkz) | https://github.com/TonghanWang/ROMA                          | ICML            | 2020     |
| Role                             | [RODE: Learning Roles to Decompose Multi-Agent Tasks](https://arxiv.org/pdf/2010.01523) | https://github.com/TonghanWang/RODE                          | ICLR            | 2021     |
| Role                             | [Scaling Multi-Agent Reinforcement Learning with Selective Parameter Sharing](http://proceedings.mlr.press/v139/christianos21a/christianos21a.pdf) | https://github.com/uoe-agents/seps                           | ICML            | 2021     |
| Opponent Modeling                | [Opponent Modeling in Deep Reinforcement Learning](https://arxiv.org/abs/1609.05559) | https://github.com/hhexiy/opponent                           | ICML            | 2016     |
| Selfish Agent                    | [M3RL: Mind-aware Multi-agent Management Reinforcement Learning](https://arxiv.org/pdf/1810.00147) | https://github.com/facebookresearch/M3RL                     | ICLR            | 2019     |
| Communication                    | [Emergence of grounded compositional language in multi-agent populations](https://ojs.aaai.org/index.php/AAAI/article/download/11492/11351) | https://github.com/bkgoksel/emergent-language                | AAAI            | 2018     |
| Communication                    | [Fully decentralized multi-agent reinforcement learning with networked agents](http://proceedings.mlr.press/v80/zhang18n/zhang18n.pdf) | https://github.com/cts198859/deeprl_network                  | ICML            | 2018     |
| Policy Gradient                  | [DOP: Off-Policy Multi-Agent Decomposed Policy Gradients](https://arxiv.org/abs/2007.12322) | https://github.com/TonghanWang/DOP                           | ICLR            | 2021     |
| Policy Gradient                  | [MAAC：Actor-Attention-Critic for Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1810.02912) | https://github.com/shariqiqbal2810/MAAC                      | ICML            | 2019     |
| Environment                      | [Emergent Complexity via Multi-Agent Competition](https://arxiv.org/pdf/1710.03748.pdfKEYWORDS: Artificial) | https://github.com/openai/multiagent-competition             | ICLR            | 2018     |
| Exploration                      | [EITI/EDTI：Influence-Based Multi-Agent Exploration](https://arxiv.org/pdf/1910.05512) | https://github.com/TonghanWang/EITI-EDTI                     | ICLR            | 2020     |
| Exploration                      | [LIIR: Learning Individual Intrinsic Reward in Multi-Agent Reinforcement Learning](http://papers.neurips.cc/paper/8691-liir-learning-individual-intrinsic-reward-in-multi-agent-reinforcement-learning.pdf) | https://github.com/yalidu/liir                               | NIPS            | 2019     |
| From Single-Agent to Multi-Agent | [MAPPO：The Surprising Effectiveness of MAPPO in Cooperative, Multi-Agent Games](https://arxiv.org/pdf/2103.01955) | https://github.com/marlbenchmark/on-policy                   |                 | 2021     |
| Diversity                        | [Q-DPP：Multi-Agent Determinantal Q-Learning](http://proceedings.mlr.press/v119/yang20i/yang20i.pdf) | https://github.com/QDPP-GitHub/QDPP                          | ICML            | 2020     |
| Ad Hoc Teamwork                  | [CollaQ：Multi-Agent Collaboration via Reward Attribution Decomposition](https://arxiv.org/pdf/2010.08531) | https://github.com/facebookresearch/CollaQ                   |                 | 2020     |
| Value Decomposition              | [NDQ: Learning Nearly Decomposable Value Functions Via Communication Minimization](https://arxiv.org/abs/1910.05366v1) | https://github.com/TonghanWang/NDQ                           | ICLR            | 2020     |
| Value Decomposition              | [QPLEX: Duplex Dueling Multi-Agent Q-Learning](https://arxiv.org/abs/2008.01062) | https://github.com/wjh720/QPLEX                              | ICLR            | 2021     |
| Self-Play                        | [TLeague: A Framework for Competitive Self-Play based Distributed Multi-Agent Reinforcement Learning](https://arxiv.org/pdf/2011.12895) | https://github.com/tencent-ailab/TLeague                     |                 | 2020     |
| Transformer                      | [UPDeT: Universal Multi-agent Reinforcement Learning via Policy Decoupling with Transformers](https://openreview.net/forum?id=v9c7hr9ADKx) | https://github.com/hhhusiyi-monash/UPDeT                     | ICLR            | 2021     |
| Sparse Reward                    | [Individual Reward Assisted Multi-Agent Reinforcement Learning](https://proceedings.mlr.press/v162/wang22ao/wang22ao.pdf) | https://github.com/MDrW/ICML2022-IRAT                        | ICML            | 2022     |
| Ad Hoc                           | [Open Ad Hoc Teamwork using Graph-based Policy Learning](http://proceedings.mlr.press/v139/rahman21a/rahman21a.pdf) | https://github.com/uoe-agents/GPL                            | ICLM            | 2021     |
| Generalization                   | [UNMAS: Multiagent Reinforcement Learningfor Unshaped Cooperative Scenarios](https://arxiv.org/pdf/2203.14477) | https://github.com/James0618/unmas                           | TNNLS           | 2021     |
| Other                            | [SIDE: State Inference for Partially Observable Cooperative Multi-Agent Reinforcement Learning](https://www.ifaamas.org/Proceedings/aamas2022/pdfs/p1400.pdf) | https://github.com/deligentfool/SIDE                         | AAMAS           | 2022     |
| Other                            | [Context-Aware Sparse Deep Coordination Graphs](https://arxiv.org/pdf/2106.02886) | https://github.com/TonghanWang/CASEC-MACO-benchmark          | ICLR            | 2022     |

# 综述

### **Recent Reviews (Since 2019)**

- [A Survey and Critique of Multiagent Deep Reinforcement Learning](https://arxiv.org/pdf/1810.05587v3)
- [An Overview of Multi-Agent Reinforcement Learning from Game Theoretical Perspective](https://arxiv.org/abs/2011.00583v2)
- [Multi-Agent Reinforcement Learning: A Selective Overview of Theories and Algorithms](https://arxiv.org/abs/1911.10635v1)
- [A Review of Cooperative Multi-Agent Deep Reinforcement Learning](https://arxiv.org/abs/1908.03963)
- [Dealing with Non-Stationarity in Multi-Agent Deep Reinforcement Learning](https://arxiv.org/abs/1906.04737)
- [A Survey of Learning in Multiagent Environments: Dealing with Non-Stationarity](https://arxiv.org/abs/1707.09183v1)
- [Deep Reinforcement Learning for Multi-Agent Systems: A Review of Challenges, Solutions and Applications](https://arxiv.org/pdf/1812.11794.pdf)
- [A Survey on Transfer Learning for Multiagent Reinforcement Learning Systems](https://www.researchgate.net/publication/330752409_A_Survey_on_Transfer_Learning_for_Multiagent_Reinforcement_Learning_Systems)

### **Other Reviews (Before 2019)**

- [If multi-agent learning is the answer, what is the question?](https://ai.stanford.edu/people/shoham/www papers/LearningInMAS.pdf)
- [Multiagent learning is not the answer. It is the question](https://core.ac.uk/download/pdf/82595758.pdf)
- [Is multiagent deep reinforcement learning the answer or the question? A brief survey](https://arxiv.org/abs/1810.05587v1) Note that [A Survey and Critique of Multiagent Deep Reinforcement Learning](https://arxiv.org/pdf/1810.05587v3) is an updated version of this paper with the same authors.
- [Evolutionary Dynamics of Multi-Agent Learning: A Survey](https://www.researchgate.net/publication/280919379_Evolutionary_Dynamics_of_Multi-Agent_Learning_A_Survey)
- (Worth reading although they're not recent reviews.)

# 环境

| **Environment** | **Paper**                                                    | **KeyWords**                     | **Code**                                           | **Accepted at** | **Year** | **Others**                                                   |
| --------------- | ------------------------------------------------------------ | -------------------------------- | -------------------------------------------------- | --------------- | -------- | ------------------------------------------------------------ |
| StarCraft       | [The StarCraft Multi-Agent Challenge](https://arxiv.org/pdf/1902.04043) |                                  | https://github.com/oxwhirl/smac                    | NIPS            | 2019     |                                                              |
| StarCraft       | [SMACv2: A New Benchmark for Cooperative Multi-Agent Reinforcement Learning](https://openreview.net/pdf?id=pcBnes02t3u) |                                  | https://github.com/oxwhirl/smacv2                  |                 | 2022     |                                                              |
| StarCraft       | [Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative Tasks](https://arxiv.org/pdf/2006.07869) |                                  | https://github.com/uoe-agents/epymarl              | NIPS            | 2021     |                                                              |
| Football        | [Google Research Football: A Novel Reinforcement Learning Environment](https://ojs.aaai.org/index.php/AAAI/article/view/5878/5734) |                                  | https://github.com/google-research/football        | AAAI            | 2020     |                                                              |
| PettingZoo      | [PettingZoo: Gym for Multi-Agent Reinforcement Learning](https://proceedings.neurips.cc/paper/2021/file/7ed2d3454c5eea71148b11d0c25104ff-Paper.pdf) |                                  | https://github.com/Farama-Foundation/PettingZoo    | NIPS            | 2021     |                                                              |
| Melting Pot     | [Scalable Evaluation of Multi-Agent Reinforcement Learning with Melting Pot](http://proceedings.mlr.press/v139/leibo21a/leibo21a.pdf) |                                  | https://github.com/deepmind/meltingpot             | ICML            | 2021     |                                                              |
| MuJoCo          | [MuJoCo: A physics engine for model-based control](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.296.6848&rep=rep1&type=pdf) |                                  | https://github.com/deepmind/mujoco                 | IROS            | 2012     |                                                              |
| MALib           | [MALib: A Parallel Framework for Population-based Multi-agent Reinforcement Learning](https://arxiv.org/pdf/2106.07551) |                                  | https://github.com/sjtu-marl/malib                 |                 | 2021     |                                                              |
| MAgent          | [MAgent: A many-agent reinforcement learning platform for artificial collective intelligence](https://ojs.aaai.org/index.php/AAAI/article/download/11371/11230) |                                  | https://github.com/Farama-Foundation/MAgent        | AAAI            | 2018     |                                                              |
| Neural MMO      | [Neural MMO: A Massively Multiagent Game Environment for Training and Evaluating Intelligent Agents](https://arxiv.org/pdf/1903.00784) |                                  | https://github.com/openai/neural-mmo               |                 | 2019     |                                                              |
| MPE             | [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://proceedings.neurips.cc/paper/2017/file/68a9750337a418a86fe06c1991a1d64c-Paper.pdf) |                                  | https://github.com/openai/multiagent-particle-envs | NIPS            | 2017     |                                                              |
| Pommerman       | [Pommerman: A multi-agent playground](https://arxiv.org/pdf/1809.07124.pdfâ€‹arxiv.org) |                                  | https://github.com/MultiAgentLearning/playground   |                 | 2018     |                                                              |
| HFO             | [Half Field Offense: An Environment for Multiagent Learning and Ad Hoc Teamwork](https://www.cse.iitb.ac.in/~shivaram/papers/hmsks_ala_2016.pdf) |                                  | https://github.com/LARG/HFO                        | AAMAS Workshop  | 2016     |                                                              |
|                 | A unified official code releasement of MARL researches made by TJU-RL-Labagents规模信用分配探索-利用平衡混合action部分观测非稳定性：自模仿+对手建模Python |                                  | https://github.com/TJU-DRL-LAB/Multiagent-RL       |                 | 2022     |                                                              |
|                 |                                                              | 自博弈强化学习环境多个博弈参与者 | https://github.com/davidADSP/SIMPLE                |                 | 2021     | ![img](https://e8bw0pe0za.feishu.cn/space/api/box/stream/download/asynccode/?code=M2FmODQ2OGU4Y2ZmY2UzMjI5ZTAwNWFhZTJhYTVkNjJfN1lyRU9ZWTFqRjg4a2RuQ09nVXNOZjRUb0xnZjZEY0hfVG9rZW46SmZUdmJEZHBkb2FaTXF4bGNyS2M1ampkblBJXzE3NDU1NDU4MzA6MTc0NTU0OTQzMF9WNA) |

# 多优化目标

## 综述类

- [Multiobjective Reinforcement Learning: A Comprehensive Overview](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6918520) 2015.

## 环境

| **Paper**                                                    | **Key Words**                                                | **Code**                                      | **Accepted at** | **Year** | **others** |      |
| ------------------------------------------------------------ | ------------------------------------------------------------ | --------------------------------------------- | --------------- | -------- | ---------- | ---- |
| [A Toolkit for Reliable Benchmarking and Research in Multi-Objective Reinforcement Learning](https://lucasalegre.github.io/assets/pdf/toolkit.pdf) | 开源的多目标强化学习算法库多目标强化学习环境MO-Gymnasium单策略和多策略方法 | https://github.com/LucasAlegre/morl-baselines |                 | 2023     |            |      |
|                                                              |                                                              |                                               |                 |          |            |      |
|                                                              |                                                              |                                               |                 |          |            |      |
|                                                              |                                                              |                                               |                 |          |            |      |

| **Paper**                                                    | **Key Words**                                                | **Code**                                     | **Accepted at** | **Year** | **others** |      |
| ------------------------------------------------------------ | ------------------------------------------------------------ | -------------------------------------------- | --------------- | -------- | ---------- | ---- |
| [A Generalized Algorithm for Multi-Objective Reinforcement Learning and Policy Adaptation](https://arxiv.org/pdf/1908.08342) | 算法框架偏好未知线性偏好优于标量化的MORL算法合理推断隐藏偏好贝尔曼方程参数化policy表示 | https://github.com/RunzheYang/MORL           | NeurIPS'19      | 2019     |            |      |
| [Lexicographic Multi-Objective Reinforcement Learning](https://arxiv.org/pdf/2212.13769) | 字典序多目标问题:目标有明确优先级 资源有限 多阶段任务可扩展性实际应用Policy-basedvalue-based单智能体 | https://github.com/lrhammond/lmorl           |                 | 2022     |            |      |
| [Multi-objective Conflict-based Search for Multi-agent Path Finding](https://arxiv.org/pdf/2101.03805.pdf). [Subdimensional Expansion for Multi-objective Multi-agent Path Finding](https://arxiv.org/pdf/2102.01353.pdf). | 帕累托最优解集基于冲突搜索                                   | https://github.com/wonderren/public_pymomapf |                 | 2021     |            |      |
| [Unifying All Species: LLM-based Hyper-Heuristics for Multi-objective Optimization](https://openreview.net/forum?id=sUywd7UhFT) | TSP多目标优化                                                |                                              |                 | 2024     |            |      |
| [Multi-objective Evolution of Heuristic Using Large Language Model](https://arxiv.org/pdf/2409.16867) | TSP,BPP多目标                                                |                                              |                 | 2024     |            |      |
| Thresholded Lexicographic Ordered Multiobjective  Reinforcement Learning | 梯度投影多智能体                                             |                                              |                 | 2024     |            |      |
| PA2D-MORL:Pareto Ascent Directional Decomposition Based Multi-Objective  Reinforcement Learning | Pareto策略集灵活性适应性结果稳定                             |                                              |                 | 2024     |            |      |
| Multi-Objective Deep Reinforcement Learning Optimisation in Autonomous Systems | 同时优化多个目标自适应服务器配置                             | https://github.com/JuanK120/RL_EWS           |                 | 2024     |            |      |
| [A practical guide to multi-objective reinforcement learning and planning](https://arxiv.org/pdf/2103.09568) | 线性标量化转为单目标 VS. 权重空间遍历存储向量值的Q值（Q-Learning）每个策略对应不同的目标组合（Pareto Q-Learning）在学习过程中动态调整偏好（Q-steering） |                                              |                 | 2021     |            |      |
| [CM3: Cooperative Multi-goal Multi-stage Multi-agent Reinforcement Learning](https://arxiv.org/pdf/1809.05188) | 智能体个体目标和集体目标平衡两阶段课程学习（平滑过渡）多目标多智能体策略梯度（信用分配机制）学习效率提升 | https://github.com/011235813/cm3             |                 | 2021     |            |      |
| MO-MIX: Multi-Objective Multi-Agent Cooperative Decision-Making With Deep Reinforcement Learning | 权重向量：平衡不同目标的重要性混合网络：整合所有智能体的局部信息，协调合作 |                                              |                 |          |            |      |

# 信用分配

## 值分解

| **Paper**                                                    | **Code**                                   | **Accepted at** | **Year** |
| ------------------------------------------------------------ | ------------------------------------------ | --------------- | -------- |
| [VDN：Value-Decomposition Networks For Cooperative Multi-Agent Learning](https://arxiv.org/pdf/1706.05296) | https://github.com/oxwhirl/pymarl          | AAMAS           | 2017     |
| [QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](http://proceedings.mlr.press/v80/rashid18a/rashid18a.pdf) | https://github.com/oxwhirl/pymarl          | ICML            | 2018     |
| [QTRAN: Learning to Factorize with Transformation for Cooperative Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1905.05408) | https://github.com/oxwhirl/pymarl          | ICML            | 2019     |
| [NDQ: Learning Nearly Decomposable Value Functions Via Communication Minimization](https://arxiv.org/abs/1910.05366v1) | https://github.com/TonghanWang/NDQ         | ICLR            | 2020     |
| [CollaQ：Multi-Agent Collaboration via Reward Attribution Decomposition](https://arxiv.org/abs/2010.08531) | https://github.com/facebookresearch/CollaQ |                 | 2020     |
| [SQDDPG：Shapley Q-Value: A Local Reward Approach to Solve Global Reward Games](https://arxiv.org/abs/1907.05707) | https://github.com/hsvgbkhgbv/SQDDPG       | AAAI            | 2020     |
| [QPD：Q-value Path Decomposition for Deep Multiagent Reinforcement Learning](http://proceedings.mlr.press/v119/yang20d/yang20d.pdf) |                                            | ICML            | 2020     |
| [Weighted QMIX: Expanding Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2006.10800) | https://github.com/oxwhirl/wqmix           | NIPS            | 2020     |
| [QTRAN++: Improved Value Transformation for Cooperative Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2006.12010v2) |                                            |                 | 2020     |
| [QPLEX: Duplex Dueling Multi-Agent Q-Learning](https://arxiv.org/abs/2008.01062) | https://github.com/wjh720/QPLEX            | ICLR            | 2021     |

## 其他方法

| **Paper**                                                    | **KeyWords**                             | **Code**                              | **Accepted at** | **Year** |      |
| ------------------------------------------------------------ | ---------------------------------------- | ------------------------------------- | --------------- | -------- | ---- |
| [COMA：Counterfactual Multi-Agent Policy Gradients](https://arxiv.org/abs/1705.08926) |                                          | https://github.com/oxwhirl/pymarl     | AAAI            | 2018     |      |
| [LiCA：Learning Implicit Credit Assignment for Cooperative Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2007.02529v2) |                                          | https://github.com/mzho7212/LICA      | NIPS            | 2020     |      |
| [Evaluating Memory and Credit Assignment in Memory-Based RL](https://arxiv.org/abs/2307.03864) | Decoupling Memory from Credit Assignment | https://github.com/twni2016/Memory-RL |                 | 2023     |      |

## 策略梯度

| **Paper**                                                    | **Code**                                   | **Accepted at** | **Year** |
| ------------------------------------------------------------ | ------------------------------------------ | --------------- | -------- |
| [MADDPG：Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/abs/1706.02275v3) | https://github.com/openai/maddpg           | NIPS            | 2017     |
| [COMA：Counterfactual Multi-Agent Policy Gradients](https://arxiv.org/abs/1705.08926) | https://github.com/oxwhirl/pymarl          | AAAI            | 2018     |
| [IPPO：Is Independent Learning All You Need in the StarCraft Multi-Agent Challenge?](https://arxiv.org/abs/2011.09533) |                                            |                 | 2020     |
| [MAPPO：The Surprising Effectiveness of MAPPO in Cooperative, Multi-Agent Games](https://arxiv.org/abs/2103.01955) | https://github.com/marlbenchmark/on-policy |                 | 2021     |
| [MAAC：Actor-Attention-Critic for Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1810.02912) | https://github.com/shariqiqbal2810/MAAC    | ICML            | 2019     |
| [DOP: Off-Policy Multi-Agent Decomposed PolicyGradients](https://arxiv.org/abs/2007.12322) | https://github.com/TonghanWang/DOP         | ICLR            | 2021     |
| [M3DDPG：Robust Multi-Agent Reinforcement Learning via Minimax Deep Deterministic Policy Gradient](https://ojs.aaai.org/index.php/AAAI/article/view/4327/4205) |                                            | AAAI            | 2019     |

# 多任务

| **Paper**                                                    | **KeyWords**                                                 | **Code**                                                     | **Accepted at** | **Year** | **Others** |      |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | --------------- | -------- | ---------- | ---- |
| [Switch Trajectory Transformer with Distributional Value Approximation for Multi-Task Reinforcement Learning](https://arxiv.org/abs/2203.07413) | Multi-Task RLSparse Reward                                   | ExpEnv: [MINIGRID](https://github.com/Farama-Foundation/gym-minigrid) |                 |          |            |      |
| [HarmoDT: Harmony Multi-Task Decision Transformer for Offline Reinforcement Learning](https://arxiv.org/abs/2405.18080) |                                                              | ExpEnv: [MetaWorld](https://github.com/Farama-Foundation/Metaworld) |                 | 2024     |            |      |
| [Elastic Decision Transformer](https://arxiv.org/abs/2307.02484) | Offline RLstitch trajectoryMulti-Task                        |                                                              |                 | 2023     |            |      |
| [Learning to Modulate pre-trained Models in RL](https://arxiv.org/abs/2306.14884) | multi-task learningcontinual learningfine-tuning             |                                                              |                 | 2023     |            |      |
|                                                              | Multi Task RL Baselines                                      | https://github.com/facebookresearch/mtrl                     |                 |          |            |      |
|                                                              | A PyTorch Library for Multi-Task Learning                    | https://github.com/median-research-group/LibMTL              |                 | 2024     |            |      |
| [Discovering Generalizable Multi-agent Coordination Skills from Multi-task Offline Data](https://openreview.net/pdf?id=53FyUAdP7d) | 有限来源的离线数据 MARL跨任务的协作未见任务泛化能力          | https://github.com/LAMDA-RL/ODIS                             |                 | 2023     |            |      |
| [Few is More: Task-Efficient Skill-Discovery for Multi-Task Offline Multi-Agent Reinforcement Learning](https://arxiv.org/pdf/2502.08985) | 避免新任务重复训练多任务离线MARL算法重构观测->评估固定动作+可变动作->正则保守动作从有限小规模源任务->强大的多任务泛化 |                                                              |                 | 2025     |            |      |

# 通信

## 带宽限制

| **Paper**                                                    | **KeyWords**         | **Code**                                    | **Accepted at** | **Year** | **Others** |
| ------------------------------------------------------------ | -------------------- | ------------------------------------------- | --------------- | -------- | ---------- |
| [SchedNet：Learning to Schedule Communication in Multi-Agent Reinforcement learning](https://arxiv.org/abs/1902.01554) |                      |                                             |                 | 2019     |            |
| [Learning Multi-agent Communication under Limited-bandwidth Restriction for Internet Packet Routing](https://arxiv.org/abs/1903.05561) |                      |                                             |                 | 2019     |            |
| [Gated-ACML：Learning Agent Communication under Limited Bandwidth by Message Pruning](https://arxiv.org/abs/1912.05304v1) |                      |                                             | AAAI            | 2020     |            |
| [Learning Efficient Multi-agent Communication: An Information Bottleneck Approach](https://arxiv.org/abs/1911.06992) |                      |                                             | ICML            | 2020     |            |
| [Coordinating Multi-Agent Reinforcement Learning with Limited Communication](http://aamas.csc.liv.ac.uk/Proceedings/aamas2013/docs/p1101.pdf) |                      |                                             | AAMAS           | 2013     |            |
| [Learning Efficient Diverse Communication for Cooperative Heterogeneous Teaming](https://www.ifaamas.org/Proceedings/aamas2022/pdfs/p1173.pdf) | 通信消息异构带宽限制 | https://github.com/CORE-Robotics-Lab/HetNet |                 | 2022     |            |

## 无带宽限制

| **Paper**                                                    | **KeyWords** | **Code**                                           | **Accepted at** | **Year** | **Others** |
| ------------------------------------------------------------ | ------------ | -------------------------------------------------- | --------------- | -------- | ---------- |
| [CommNet：Learning Multiagent Communication with Backpropagation](https://arxiv.org/abs/1605.07736) |              | https://github.com/facebookarchive/CommNet         | NIPS            | 2016     |            |
| [BiCNet：Multiagent Bidirectionally-Coordinated Nets: Emergence of Human-level Coordination in Learning to Play StarCraft Combat Games](https://arxiv.org/abs/1703.10069) |              | https://github.com/Coac/CommNet-BiCnet             |                 | 2017     |            |
| [VAIN: Attentional Multi-agent Predictive Modeling](https://arxiv.org/abs/1706.06122) |              |                                                    | NIPS            | 2017     |            |
| [IC3Net：Learning when to Communicate at Scale in Multiagent Cooperative and Competitive Tasks](https://arxiv.org/abs/1812.09755) |              | https://github.com/IC3Net/IC3Net                   |                 | 2018     |            |
| [VBC：Efficient Communication in Multi-Agent Reinforcement Learning via Variance Based Control](https://arxiv.org/abs/1909.02682v1) |              |                                                    | NIPS            | 2019     |            |
| [Graph Convolutional Reinforcement Learning for Multi-Agent Cooperation](https://arxiv.org/abs/1810.09202v1) |              |                                                    |                 | 2018     |            |
| [NDQ：Learning Nearly Decomposable Value Functions Via Communication MinimizationNDQ: Learning Nearly Decomposable Value Functions Via Communication Minimization](https://arxiv.org/abs/1910.05366v1) |              | https://github.com/TonghanWang/NDQ                 | ICLR            | 2020     |            |
| [RIAL/RIDL：Learning to Communicate with Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1605.06676) |              | https://github.com/iassael/learning-to-communicate | NIPS            | 2016     |            |
| [ATOC：Learning Attentional Communication for Multi-Agent Cooperation](https://arxiv.org/abs/1805.07733) |              |                                                    | NIPS            | 2018     |            |
| [Fully decentralized multi-agent reinforcement learning with networked agents](http://proceedings.mlr.press/v80/zhang18n/zhang18n.pdf) |              | https://github.com/cts198859/deeprl_network        | ICML            | 2018     |            |
| [TarMAC: Targeted Multi-Agent Communication](http://proceedings.mlr.press/v97/das19a/das19a.pdf) |              |                                                    | ICML            | 2019     |            |

## 未分

| **Paper**                                                    | **Key Words**                                                | **Code**                                       | **Accepted at** | **Year** | **Others**                                                   |      |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ---------------------------------------------- | --------------- | -------- | ------------------------------------------------------------ | ---- |
|                                                              |                                                              |                                                |                 |          |                                                              |      |
| [Responsive Regulation of Dynamic UAV Communication Networks Based on Deep Reinforcement Learning](https://arxiv.org/pdf/2108.11012) | 无人机（UAV）通信网络动态调控异步DDPGPython                  | https://github.com/ducmngx/DDPG-UAV-Efficiency |                 | 2021     | ![img](https://e8bw0pe0za.feishu.cn/space/api/box/stream/download/asynccode/?code=M2M1NjVmMzdmNjgyNTViZDQ3YTRiMDJiYWYxZmQxY2FfNnBEb25RaGl1WUR4NlJkcWJDQk9hUmV0dG1rZE0ydU1fVG9rZW46UjRtS2JJcVRYbzRXZWt4ZThkR2Mzak1hbnVmXzE3NDU1NDU4MzA6MTc0NTU0OTQzMF9WNA) |      |
| [eQMARL: Entangled Quantum Multi-Agent Reinforcement Learning for Distributed Cooperation over Quantum Channels](https://arxiv.org/pdf/2405.17486) | 分布式多智能体强化学习信息共享量子通道收敛更快 效果更好 减少计算负担 | https://github.com/news-vt/eqmarl              |                 | 2025     |                                                              |      |
| [Learning to Communicate Through Implicit Communication Channels](https://arxiv.org/pdf/2411.01553) | 隐式通信协议（ICP）框架更高效                                |                                                |                 | 2024     |                                                              |      |
| [Scaling Large Language Model-based Multi-Agent Collaboration](https://arxiv.org/pdf/2406.07155) | LLM协作涌现通信模式有向无环图1000个智能体                    | https://github.com/OpenBMB/ChatDev             |                 | 2024     |                                                              |      |

# 涌现

| **Paper**                                                    | **KeyWords**                              | **Code**                                                     | **Accepted at**    | **Year** |      |
| ------------------------------------------------------------ | ----------------------------------------- | ------------------------------------------------------------ | ------------------ | -------- | ---- |
| [Multiagent Cooperation and Competition with Deep Reinforcement Learning](https://arxiv.org/abs/1511.08779v1) |                                           |                                                              | PloS one           | 2017     |      |
| [Multi-agent Reinforcement Learning in Sequential Social Dilemmas](https://arxiv.org/abs/1702.03037) |                                           |                                                              |                    | 2017     |      |
| [Emergent preeminence of selfishness: an anomalous Parrondo perspective](https://kanghaocheong.files.wordpress.com/2020/02/koh-cheong2019_article_emergentpreeminenceofselfishne.pdf) |                                           |                                                              | Nonlinear Dynamics | 2019     |      |
| [Emergent Coordination Through Competition](https://arxiv.org/abs/1902.07151v2) |                                           |                                                              |                    | 2019     |      |
| [Biases for Emergent Communication in Multi-agent Reinforcement Learning](https://arxiv.org/abs/1912.05676) |                                           |                                                              | NIPS               | 2019     |      |
| [Towards Graph Representation Learning in Emergent Communication](https://arxiv.org/abs/2001.09063) |                                           |                                                              |                    | 2020     |      |
| [Emergent Tool Use From Multi-Agent Autocurricula](https://arxiv.org/abs/1909.07528) |                                           | https://github.com/openai/multi-agent-emergence-environments | ICLR               | 2020     |      |
| [On Emergent Communication in Competitive Multi-Agent Teams](https://arxiv.org/abs/2003.01848) |                                           |                                                              | AAMAS              | 2020     |      |
| [QED：Quasi-Equivalence Discovery for Zero-Shot Emergent Communication](https://arxiv.org/abs/2103.08067) |                                           |                                                              |                    | 2021     |      |
| [Incorporating Pragmatic Reasoning Communication into Emergent Language](https://arxiv.org/abs/2006.04109) |                                           |                                                              | NIPS               | 2020     |      |
| [Scaling Large Language Model-based Multi-Agent Collaboration](https://arxiv.org/pdf/2406.07155) | LLM协作涌现通信模式有向无环图1000个智能体 | https://github.com/OpenBMB/ChatDev                           |                    | 2024     |      |

# 对手建模

| **Paper**                                                    | **Code**                           | **Accepted at**                                         | **Year** |
| ------------------------------------------------------------ | ---------------------------------- | ------------------------------------------------------- | -------- |
| [Bayesian Opponent Exploitation in Imperfect-Information Games](https://arxiv.org/abs/1603.03491v1) |                                    | IEEE Conference on Computational Intelligence and Games | 2018     |
| [LOLA：Learning with Opponent-Learning Awareness](https://arxiv.org/abs/1709.04326) |                                    | AAMAS                                                   | 2018     |
| [Variational Autoencoders for Opponent Modeling in Multi-Agent Systems](https://arxiv.org/abs/2001.10829) |                                    |                                                         | 2020     |
| [Stable Opponent Shaping in Differentiable Games](https://arxiv.org/abs/1811.08469) |                                    |                                                         | 2018     |
| [Opponent Modeling in Deep Reinforcement Learning](https://arxiv.org/abs/1609.05559) | https://github.com/hhexiy/opponent | ICML                                                    | 2016     |
| [Game Theory-Based Opponent Modeling in Large Imperfect-Information Games](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.385.6032&rep=rep1&type=pdf) |                                    | AAMAS                                                   | 2011     |
| [Agent Modelling under Partial Observability for Deep Reinforcement Learning](https://proceedings.neurips.cc/paper/2021/file/a03caec56cd82478bf197475b48c05f9-Paper.pdf) |                                    | NIPS                                                    | 2021     |

# 博弈论

| **Paper**                                                    | **KeyWords**                                                 | **Code**                                                     | **Accepted at**    | **Year** | **Others**                                                   |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------ | -------- | ------------------------------------------------------------ |
| [α-Rank: Multi-Agent Evaluation by Evolution](https://arxiv.org/abs/1903.01373) |                                                              |                                                              | Scientific reports | 2019     |                                                              |
| [α^α -Rank: Practically Scaling α-Rank through Stochastic Optimisation](https://arxiv.org/abs/1909.11628) |                                                              |                                                              | AAMAS              | 2020     |                                                              |
| [A Game Theoretic Framework for Model Based Reinforcement Learning](https://arxiv.org/abs/2004.07804) |                                                              |                                                              | ICML               | 2020     |                                                              |
| [Fictitious Self-Play in Extensive-Form Games](http://proceedings.mlr.press/v37/heinrich15.pdf) |                                                              |                                                              | ICML               | 2015     |                                                              |
| [Combining Deep Reinforcement Learning and Search for Imperfect-Information Games](https://arxiv.org/pdf/2007.13544) |                                                              |                                                              | NIPS               | 2020     |                                                              |
| [Real World Games Look Like Spinning Tops](https://arxiv.org/pdf/2004.09468) |                                                              |                                                              | NIPS               | 2020     |                                                              |
| [PSRO: A Unified Game-Theoretic Approach to Multiagent Reinforcement Learning](https://arxiv.org/pdf/1711.00832) |                                                              |                                                              | NIPS               | 2017     |                                                              |
| [Pipeline PSRO: A Scalable Approach for Finding Approximate Nash Equilibria in Large Games](https://arxiv.org/pdf/2006.08555) |                                                              |                                                              | NIPS               | 2020     |                                                              |
| [A Game-Theoretic Model and Best-Response Learning Method for Ad Hoc Coordination in Multiagent Systems](https://arxiv.org/pdf/1506.01170) |                                                              |                                                              | AAMAS              | 2013     |                                                              |
| [Neural Replicator Dynamics: Multiagent Learning via Hedging Policy Gradients](http://www.ifaamas.org/Proceedings/aamas2020/pdfs/p492.pdf) |                                                              |                                                              | AAMAS              | 2020     |                                                              |
| [ASP: Learn a Universal Neural Solver!](https://arxiv.org/abs/2303.00466) |                                                              | https://github.com/LOGO-CUHKSZ/ASP                           | IEEE               | 2023     |                                                              |
|                                                              | 自博弈代码框架Python                                         | https://github.com/davidADSP/SIMPLE                          |                    | 2021     | ![img](https://e8bw0pe0za.feishu.cn/space/api/box/stream/download/asynccode/?code=NmIxY2I1NjA3YmJiZDNjMzdmM2MxZTlkZjdmZjFhZGRfckowYXBNOGNIODhERmdWaTQ0UlQ4dk9rN1l6a2hSRzBfVG9rZW46RXhMTmJFZUJob2toQ1l4eThIQmNDZ3FYblVQXzE3NDU1NDU4MzA6MTc0NTU0OTQzMF9WNA) |
| [TimeChamber: A Massively Parallel Large Scale Self-Play Framework](https://forums.developer.nvidia.com/t/timechamber-a-massively-parallel-large-scale-self-play-framework/226567) | 大规模并行自我对战框架Python                                 | https://github.com/inspirai/TimeChamber                      |                    | 2022     |                                                              |
|                                                              | 复现与多智能体博弈相关的论文                                 | https://github.com/BaoyiCui/MAS-Game                         |                    |          |                                                              |
| [Minimizing Weighted Counterfactual Regret with Optimistic Online Mirror Descent](https://arxiv.org/pdf/2404.13891) | 反事实遗憾最小化（Counterfactual Regret Minimization, CFR）不完全信息博弈快速收敛 | https://github.com/rpSebastian/PDCFRPlus                     |                    | 2024     |                                                              |
| [Dynamic Discounted Counterfactual Regret Minimization](https://openreview.net/pdf?id=6PbvbLyqT6) | 第一个使用动态的、自动学习的方案来对先前迭代进行折扣的均衡求解框架泛化收敛速度 | https://github.com/rpSebastian/DDCFR                         |                    | 2024     |                                                              |
|                                                              | Awesome Game AI materials of Multi-Agent Reinforcement Learning | https://github.com/datamllab/awesome-game-ai                 |                    |          |                                                              |
| [Pipeline PSRO: A Scalable Approach for Finding Approximate Nash Equilibria in Large Games](https://arxiv.org/pdf/2006.08555) | 零和不完全信息博弈收敛速度快                                 | https://github.com/JBLanier/pipeline-psro                    |                    | 2020     |                                                              |
| [Neural Auto-Curricula](https://arxiv.org/pdf/2106.02745)    | 神经自动课程自动：选择对手策略 + 寻找最佳响应元梯度下降通用MARL算法可扩展性 | https://github.com/waterhorse1/NAC                           |                    | 2021     | ![img](https://e8bw0pe0za.feishu.cn/space/api/box/stream/download/asynccode/?code=ZjExN2MxNzU5ZDM1NDZlMzFmYTQ1NWM4OWIyMGZkYjJfdW83ajFhcXc2cHY0ZFlkdDVRRHVweTlRMzBlVG5mZG9fVG9rZW46RzNCaGJXTlFTb0pwZ2R4cTV0QmM3WnhSbnBoXzE3NDU1NDU4MzA6MTc0NTU0OTQzMF9WNA) |
| [Temporal Induced Self-Play for Stochastic Bayesian Games](https://arxiv.org/pdf/2108.09444) | 动态博弈问题基于强化学习的算法框架基于策略梯度可扩展性       | https://github.com/laonahongchen/Temporal-Induced-Self-Play-for-Stochastic-Bayesian-Games |                    | 2020     |                                                              |

# 分层

| **Paper**                                                    | **Code** | **Accepted at** | **Year** |
| ------------------------------------------------------------ | -------- | --------------- | -------- |
| [Hierarchical multi-agent reinforcement learning](https://apps.dtic.mil/sti/pdfs/ADA440418.pdf) |          | AAMAS           | 2006     |
| [Hierarchical Cooperative Multi-Agent Reinforcement Learning with Skill Discovery](https://arxiv.org/pdf/1912.03558) |          | AAMAS           | 2020     |
| [Hierarchical Critics Assignment for Multi-agent Reinforcement Learning](https://arxiv.org/pdf/1902.03079) |          |                 | 2019     |
| [Hierarchical Reinforcement Learning for Multi-agent MOBA Game](https://arxiv.org/pdf/1901.08004) |          |                 | 2019     |
| [Hierarchical Deep Multiagent Reinforcement Learning with Temporal Abstraction](https://arxiv.org/pdf/1809.09332) |          |                 | 2018     |
| [HAMA：Multi-Agent Actor-Critic with Hierarchical Graph Attention Network](https://ojs.aaai.org/index.php/AAAI/article/download/6214/6070) |          | AAAI            | 2020     |

# 角色

| **Paper**                                                    | **KeyWords**                                                 | **Code**                            | **Accepted at** | **Year** |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ----------------------------------- | --------------- | -------- |
| [ROMA: Multi-Agent Reinforcement Learning with Emergent Roles](https://openreview.net/pdf?id=RQP2wq-dbkz) |                                                              | https://github.com/TonghanWang/ROMA | ICML            | 2020     |
| [RODE: Learning Roles to Decompose Multi-Agent Tasks](https://arxiv.org/pdf/2010.01523) |                                                              | https://github.com/TonghanWang/RODE | ICLR            | 2021     |
| [Scaling Multi-Agent Reinforcement Learning with Selective Parameter Sharing](http://proceedings.mlr.press/v139/christianos21a/christianos21a.pdf) |                                                              | https://github.com/uoe-agents/seps  | ICML            | 2021     |
| [MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework](https://arxiv.org/pdf/2308.00352) | 元编程框架：流水线范式智能体角色任务分解：子任务更连贯的解决方案 | https://github.com/geekan/MetaGPT   |                 | 2023     |

# 大规模

| **Paper**                                                    | **Key Words**                                                | **Code**                                          | **Accepted at** | **Year** |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------- | --------------- | -------- |
| [From Few to More: Large-Scale Dynamic Multiagent Curriculum Learning](https://arxiv.org/abs/1909.02790) |                                                              | https://github.com/starry-sky6688/MARL-Algorithms | AAAI            | 2020     |
| [PooL: Pheromone-inspired Communication Framework for Large Scale Multi-Agent Reinforcement Learning](https://arxiv.org/pdf/2202.09722) |                                                              |                                                   |                 | 2022     |
| [Factorized Q-learning for large-scale multi-agent systems](https://dl.acm.org/doi/pdf/10.1145/3356464.3357707?casa_token=CNK3OslP6hkAAAAA:yZFMOmNQB1iasPqxmA6DYDIFe79RdMqUu_8Y7sGASsPNQ3u4o0UkAcqwMTahAwSUcDuh5r6NvSAyig) |                                                              |                                                   | ICDAI           | 2019     |
| [EPC：Evolutionary Population Curriculum for Scaling Multi-Agent Reinforcement Learning](https://arxiv.org/pdf/2003.10423) |                                                              | https://github.com/qian18long/epciclr2020         | ICLR            | 2020     |
| [Mean Field Multi-Agent Reinforcement Learning](http://proceedings.mlr.press/v80/yang18d/yang18d.pdf) |                                                              |                                                   | ICML            | 2018     |
| [A Study of AI Population Dynamics with Million-agent Reinforcement Learning](https://arxiv.org/pdf/1709.04511) |                                                              |                                                   | AAMAS           | 2018     |
| [Plan Better Amid Conservatism: Offline Multi-Agent Reinforcement Learning with Actor Rectification](https://proceedings.mlr.press/v162/pan22a/pan22a.pdf) | 离线RL智能体数量增加 陷入局部最优方法：带演员修正的离线多智能体强化学习 | https://github.com/ling-pan/OMAR                  |                 | 2022     |

# 即兴协作

| **Paper**                                                    | **Key Words**                                                | **Code**                                   | **Accepted at** | **Year** |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------ | --------------- | -------- |
| [CollaQ：Multi-Agent Collaboration via Reward Attribution Decomposition](https://arxiv.org/pdf/2010.08531) |                                                              | https://github.com/facebookresearch/CollaQ |                 | 2020     |
| [A Game-Theoretic Model and Best-Response Learning Method for Ad Hoc Coordination in Multiagent Systems](https://arxiv.org/pdf/1506.01170) |                                                              |                                            | AAMAS           | 2013     |
| [Half Field Offense: An Environment for Multiagent Learning and Ad Hoc Teamwork](https://www.cse.iitb.ac.in/~shivaram/papers/hmsks_ala_2016.pdf) |                                                              | https://github.com/LARG/HFO                | AAMAS Workshop  | 2016     |
| [Open Ad Hoc Teamwork using Graph-based Policy Learning](http://proceedings.mlr.press/v139/rahman21a/rahman21a.pdf) |                                                              | https://github.com/uoe-agents/GPL          | ICLM            | 2021     |
| [A Survey of Ad Hoc Teamwork: Definitions, Methods, and Open Problems](https://arxiv.org/pdf/2202.10450) |                                                              |                                            |                 | 2022     |
| [Towards open ad hoc teamwork using graph-based policy learning](http://proceedings.mlr.press/v139/rahman21a/rahman21a.pdf) |                                                              |                                            | ICML            | 2021     |
| [Learning with generated teammates to achieve type-free ad-hoc teamwork](https://www.ijcai.org/proceedings/2021/0066.pdf) |                                                              |                                            | IJCAI           | 2021     |
| [Online ad hoc teamwork under partial observability](https://openreview.net/pdf?id=18Ys0-PzyPI) |                                                              |                                            | ICLR            | 2022     |
| [Discovering Generalizable Multi-agent Coordination Skills from Multi-task Offline Data](https://openreview.net/pdf?id=53FyUAdP7d) | 有限来源的离线数据 MARL跨任务的协作未见任务泛化能力          | https://github.com/LAMDA-RL/ODIS           |                 | 2023     |
| [Few is More: Task-Efficient Skill-Discovery for Multi-Task Offline Multi-Agent Reinforcement Learning](https://arxiv.org/pdf/2502.08985) | 避免新任务重复训练多任务离线MARL算法重构观测->评估固定动作+可变动作->正则保守动作从有限小规模源任务->强大的多任务泛化 |                                            |                 | 2025     |
| [MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework](https://arxiv.org/pdf/2308.00352) | 元编程框架：流水线范式智能体角色任务分解：子任务更连贯的解决方案 | https://github.com/geekan/MetaGPT          |                 | 2023     |

# 进化算法

## 综述类

- [Bridging Evolutionary Algorithms and Reinforcement Learning: A Comprehensive Survey on Hybrid Algorithms](https://arxiv.org/pdf/2401.11963v4)
- Combining evolution and deep reinforcement learning for policy search: a survey
- Deep reinforcement learning versus evolution strategies: A comparative survey
- A survey on evolutionary reinforcement learning algorithms
- Reinforcement learning versus evolutionary computation: A survey on hybrid algorithms
- Evolutionary computation and the reinforcement learning problem
- Evolutionary reinforcement learning: A survey

| **Types**                                                    | **Paper**                                                    | **Key Words**                                                | **Code**                                                     | **Accepted at**          | **Year** | **Others**                                                   |      |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------ | -------- | ------------------------------------------------------------ | ---- |
|                                                              | [Socialized Learning: Making Each Other Better Through Multi-Agent Collaboration](https://openreview.net/pdf?id=aaeJpJw5Ur) | 社会化学习（Socialized Learning, SL）集体协作 互惠利他模块   | https://github.com/yxjdarren/SL                              |                          | 2024     |                                                              |      |
|                                                              | [RACE: Improve Multi-Agent Reinforcement Learning with Representation Asymmetry and Collaborative Evolution](https://proceedings.mlr.press/v202/li23i/li23i.pdf) | MARL+演化算法+表征学习                                       | https://github.com/yeshenpy/RACE                             |                          | 2023     |                                                              |      |
|                                                              | [MALib: A Parallel Framework for Population-based Multi-agent Reinforcement Learning](https://arxiv.org/pdf/2106.07551) | 基于种群的多智能体强化学习自动课程学习可扩展速度比RLlib快5倍比OpenSpiel至少快3倍 | https://github.com/sjtu-marl/malib                           |                          | 2021     |                                                              |      |
|                                                              | [EvoRainbow: Combining Improvements in Evolutionary Reinforcement Learning for Policy Search](https://openreview.net/pdf?id=75Hes6Zse4) | 进化算法（EAs）和强化学习（RL）结合机制探索5种Python         | https://github.com/yeshenpy/EvoRainbow                       |                          | 2024     | ![img](https://e8bw0pe0za.feishu.cn/space/api/box/stream/download/asynccode/?code=ODIyYTEzNjgwYjY2MDYwZDQzNGI0NTU4NmIxMmUwN2RfUTlZbVNVcU02VGh1TEhsdGJEVUtZU2xjUmVnVnpiRmZfVG9rZW46TUxCT2JpSlNib3JIS3R4YUhCd2NCOTRvbjFjXzE3NDU1NDU4MzA6MTc0NTU0OTQzMF9WNA) |      |
|                                                              |                                                              |                                                              |                                                              |                          |          |                                                              |      |
| 1.**EA辅助RL****-参数搜索**                                  | [Reinforcement Learning beyond The Bellman Equation: Exploring Critic Objectives using Evolution](https://direct.mit.edu/isal/proceedings/isal2020/32/441/98464) |                                                              | https://github.com/ajleite/RLBeyondBellman                   |                          | 2020     |                                                              |      |
| [Genetic Soft Updates for Policy Evolution in Deep Reinforcement Learning](https://openreview.net/forum?id=TGFO0DbD_pk) |                                                              |                                                              |                                                              | 2021                     |          |                                                              |      |
| [Improving Deep Policy Gradients with Value Function Search](https://openreview.net/forum?id=6qZC7pfenQm) |                                                              |                                                              |                                                              | 2023                     |          |                                                              |      |
|                                                              |                                                              |                                                              |                                                              |                          |          |                                                              |      |
|                                                              |                                                              |                                                              |                                                              |                          |          |                                                              |      |
| 1.**EA辅助RL-Action搜索**                                    | [Scalable deep reinforcement learning for vision-based robotic manipulation](https://proceedings.mlr.press/v87/kalashnikov18a) |                                                              | https://github.com/quantumiracle/QT_Opt                      |                          | 2018     |                                                              |      |
| [RL4RealLife Workshop Q-learning for continuous actions with cross-entropy guided policies](https://arxiv.org/abs/1903.10605) |                                                              |                                                              |                                                              | 2019                     |          |                                                              |      |
| [Evolutionary Action Selection for Gradient-based Policy Learning](https://arxiv.org/abs/2201.04286) |                                                              |                                                              |                                                              | 2022                     |          |                                                              |      |
| [Soft Actor-Critic with Cross-entropy Policy Optimization](https://arxiv.org/abs/2112.11115) |                                                              | https://github.com/wcgcyx/SAC-CEPO                           |                                                              | 2021                     |          |                                                              |      |
| [GRAC: Self-guided and Self-regularized Actor-critic](https://arxiv.org/abs/2009.08973) |                                                              | https://github.com/stanford-iprl-lab/GRAC                    |                                                              | 2021                     |          |                                                              |      |
| [Plan better amid conservatism: Offline multi-agent reinforcement learning with actor rectification](https://arxiv.org/abs/2111.11188) |                                                              | https://github.com/ling-pan/OMAR                             |                                                              | 2022                     |          |                                                              |      |
| [Deep Multi-agent Reinforcement Learning for Decentralized Continuous Cooperative Control](https://beipeng.github.io/files/2003.06709.pdf) |                                                              | https://github.com/oxwhirl/comix                             |                                                              | 2020                     |          |                                                              |      |
|                                                              |                                                              |                                                              |                                                              |                          |          |                                                              |      |
|                                                              |                                                              |                                                              |                                                              |                          |          |                                                              |      |
| 1.**EA辅助RL-超参优化**                                      | [Online Meta-learning by Parallel Algorithm Competition](https://arxiv.org/abs/1702.07490) |                                                              |                                                              |                          | 2018     |                                                              |      |
| [Population Based Training of Neural Networks](https://arxiv.org/abs/1711.09846) |                                                              | https://github.com/voiler/PopulationBasedTraining            |                                                              | 2017                     |          |                                                              |      |
| [Sample-efficient Automated Deep Reinforcement Learning](https://arxiv.org/abs/2009.01555) |                                                              | https://github.com/automl/SEARL                              |                                                              | 2021                     |          |                                                              |      |
| [GA+DDPG+HER: Genetic Algorithm-based Function Optimizer in Deep Reinforcement Learning for Robotic Manipulation Tasks](https://arxiv.org/abs/2203.00141) |                                                              | https://github.com/aralab-unr/ga-drl-aubo-ara-lab            |                                                              | 2022                     |          |                                                              |      |
| [Towards Automatic Actor-critic Solutions to Continuous Control](https://arxiv.org/abs/2106.08918) |                                                              | https://github.com/jakegrigsby/deep_control                  |                                                              | 2021                     |          |                                                              |      |
| [Online Hyper-parameter Tuning in Offpolicy Learning via Evolutionary Strategies](https://arxiv.org/abs/2006.07554) |                                                              |                                                              |                                                              | 2020                     |          |                                                              |      |
| 1.**EA辅助RL-其他**                                          | [Evolving Reinforcement Learning Algorithms](https://arxiv.org/abs/2101.03958) |                                                              | https://github.com/google/brain_autorl/tree/main/evolving_rl |                          | 2021     |                                                              |      |
| [Discovered Policy Optimisation](https://arxiv.org/abs/2210.05639) |                                                              | https://github.com/luchris429/discovered-policy-optimisation |                                                              | 2022                     |          |                                                              |      |
| [Discovering Temporally-Aware Reinforcement Learning Algorithms](https://arxiv.org/abs/2402.05828) | 时序感知RL？                                                 | https://github.com/EmptyJackson/groove                       |                                                              | 2024                     |          |                                                              |      |
| [Behaviour Distillation](https://openreview.net/forum?id=qup9xD8mW4) |                                                              | https://github.com/FLAIROx/behaviour-distillation            |                                                              | 2024                     |          |                                                              |      |
| [Adversarial Cheap Talk](https://arxiv.org/abs/2211.11030)   |                                                              | https://github.com/luchris429/adversarial-cheap-talk         |                                                              | 2023                     |          |                                                              |      |
| [PNS: Population-guided Novelty Search for Reinforcement Learning in Hard Exploration Environments](https://arxiv.org/abs/1811.10264) |                                                              |                                                              |                                                              | 2021                     |          |                                                              |      |
| [Go explore: A New Approach for Hard-exploration Problems](https://arxiv.org/abs/1901.10995) |                                                              | https://github.com/uber-research/go-explore                  | Nature                                                       | 2021                     |          |                                                              |      |
| [Genetic-gated Networks for Deep Reinforcement Learning](https://arxiv.org/abs/1903.01886) |                                                              |                                                              |                                                              | 2018                     |          |                                                              |      |
| [Evo-rl: Evolutionary-driven Reinforcement Learning](https://arxiv.org/abs/2007.04725) |                                                              |                                                              |                                                              | 2021                     |          |                                                              |      |
| [Robust Multi-agent Coordination via Evolutionary Generation of Auxiliary Adversarial Attackers](https://arxiv.org/abs/2305.05909) |                                                              | https://github.com/zzq-bot/ROMANCE                           |                                                              | 2023                     |          |                                                              |      |
| [Communication-robust Multiagent Learning by Adaptable Auxiliary Multi-agent Adversary Generation](https://arxiv.org/abs/2305.05116) |                                                              |                                                              |                                                              | 2023                     |          |                                                              |      |
| [Evolutionary Population Curriculum for Scaling Multi-agent Reinforcement Learning](https://arxiv.org/abs/2003.10423) |                                                              | https://github.com/qian18long/epciclr2020                    |                                                              | 2020                     |          |                                                              |      |
| [MAPPER: Multi-agent Path Planning with Evolutionary Reinforcement Learning in Mixed Dynamic Environments](https://arxiv.org/abs/2007.15724) |                                                              |                                                              |                                                              | 2020                     |          |                                                              |      |
| **2.RL辅助EA-种群初始化**                                    | [Symbolic Regression Via Neural-guided Genetic Programming Population Seeding](https://arxiv.org/pdf/2111.00053.pdf) |                                                              | https://github.com/dso-org/deep-symbolic-optimization        |                          | 2021     |                                                              |      |
| [Rule-based Reinforcement Learning Methodology To Inform Evolutionary Algorithms For Constrained Optimization Of Engineering Applications](https://www.sciencedirect.com/science/article/abs/pii/S095070512100099X) |                                                              | https://github.com/mradaideh/neorl                           |                                                              | 2021                     |          |                                                              |      |
| [Deepaco: Neuralenhanced Ant Systems For Combinatorial Optimization](https://arxiv.org/abs/2309.14032) |                                                              | https://github.com/henry-yeh/DeepACO                         |                                                              | 2023                     |          |                                                              |      |
| **2.RL辅助EA-种群评估**                                      | [ERL-Re2: Efficient Evolutionary Reinforcement Learning with Shared State Representation and Individual Policy Representation](https://arxiv.org/abs/2210.17375) |                                                              | https://github.com/yeshenpy/ERL-Re2                          |                          | 2023     |                                                              |      |
| [A Surrogate-Assisted Controller for Expensive Evolutionary Reinforcement Learning](https://arxiv.org/abs/2201.00129) |                                                              | https://github.com/Yuxing-Wang-THU/Surrogate-assisted-ERL    |                                                              |                          |          |                                                              |      |
| [PGPS: Coupling Policy Gradient with Population-based Search](https://openreview.net/forum?id=PeT5p3ocagr) |                                                              | https://github.com/NamKim88/PGPS/blob/master/Main.py         |                                                              | 2021                     |          |                                                              |      |
| **2.RL辅助EA-变异操作**                                      | [Policy Optimization By Genetic Distillation](https://arxiv.org/abs/1711.01012) |                                                              | https://www.catalyzex.com/paper/policy-optimization-by-genetic-distillation/code |                          | 2018     |                                                              |      |
| [Guiding Evolutionary Strategies With Off-policy Actor-critic](https://robintyh1.github.io/papers/Tang2021CEMACER.pdf) |                                                              |                                                              |                                                              | 2021                     |          |                                                              |      |
| [Population Based Reinforcement Learning](https://ieeexplore.ieee.org/document/9660084) |                                                              | https://github.com/jjccero/pbrl                              |                                                              | 2021                     |          |                                                              |      |
| [Efficient Novelty Search Through Deep Reinforcement Learning](https://ieeexplore.ieee.org/document/9139203) |                                                              | https://github.com/shilx001/NoveltySearch_Improvement        |                                                              | 2020                     |          |                                                              |      |
| [Diversity Evolutionary Policy Deep Reinforcement Learning](https://www.hindawi.com/journals/cin/2021/5300189/) |                                                              |                                                              |                                                              | 2021                     |          |                                                              |      |
| [QD-RL: Efficient Mixing Of Quality And Diversity In Reinforcement Learning](https://www.researchgate.net/publication/342198149_QD-RL_Efficient_Mixing_of_Quality_and_Diversity_in_Reinforcement_Learning), |                                                              | https://openreview.net/forum?id=5Dl1378QutR                  |                                                              | 2020                     |          |                                                              |      |
| [Policy Gradient Assisted Map-elites](https://www.semanticscholar.org/paper/Policy-gradient-assisted-MAP-Elites-Nilsson-Cully/67038237383a8f4802a9595636a6fb73f748dc5b) |                                                              | https://github.com/ollebompa/PGA-MAP-Elites                  |                                                              | 2021                     |          |                                                              |      |
| [Approximating Gradients For Differentiable Quality Diversity In Reinforcement Learning](https://arxiv.org/abs/2202.03666) |                                                              | https://github.com/icaros-usc/dqd-rl                         |                                                              | 2022                     |          |                                                              |      |
| [Sample-efficient Quality-diversity By Cooperative Coevolution](https://openreview.net/forum?id=JDud6zbpFv) |                                                              | https://openreview.net/forum?id=JDud6zbpFv                   |                                                              | 2024                     |          |                                                              |      |
| [Neuroevolution is a Competitive Alternative to Reinforcement Learning for Skill Discovery](https://openreview.net/forum?id=6BHlZgyPOZY) |                                                              | https://github.com/instadeepai/qd-skill-discovery-benchmark  |                                                              | 2023                     |          |                                                              |      |
| [Approximating Gradients for Differentiable Quality Diversity in Reinforcement Learning](https://arxiv.org/pdf/2202.03666.pdf) |                                                              | https://github.com/icaros-usc/dqd-rl                         |                                                              | 2022                     |          |                                                              |      |
| [CEM-RL: Combining evolutionary and gradient-based methods for policy search](https://arxiv.org/abs/1810.01222) |                                                              | https://github.com/apourchot/CEM-RL                          |                                                              | 2019                     |          |                                                              |      |
|                                                              |                                                              |                                                              |                                                              |                          |          |                                                              |      |
| **2.RL辅助EA-超参配置**                                      | [Reinforcement learning for online control of evolutionary algorithms](https://link.springer.com/chapter/10.1007/978-3-540-69868-5_10) |                                                              |                                                              |                          | 2006     |                                                              |      |
| [Learning step-size adaptation in CMA-ES](https://ml.informatik.uni-freiburg.de/wp-content/uploads/papers/20-PPSN-LTO-CMA.pdf) |                                                              | https://github.com/automl/LTO-CMA                            |                                                              | 2020                     |          |                                                              |      |
| [Dynamic algorithm configuration: Foundation of a new meta-algorithmic framework](https://ecai2020.eu/papers/1237_paper.pdf) | 框架                                                         | https://github.com/automl/DAC                                |                                                              | 2020                     |          |                                                              |      |
| [Variational reinforcement learning for hyper-parameter tuning of adaptive evolutionary algorithm](https://www.researchgate.net/publication/365582495_Variational_Reinforcement_Learning_for_Hyper-Parameter_Tuning_of_Adaptive_Evolutionary_Algorithm) |                                                              |                                                              |                                                              | 2022                     |          |                                                              |      |
| [Controlling sequential hybrid evolutionary algorithm by q-learning](https://ieeexplore.ieee.org/document/10035716/) |                                                              | https://github.com/xiaomeiabc/Controlling-Sequential-Hybrid-Evolutionary-Algorithm-by-Q-Learning |                                                              | 2023                     |          |                                                              |      |
| [Multiagent dynamic algorithm configuration](https://arxiv.org/abs/2210.06835) |                                                              | https://github.com/lamda-bbo/madac                           |                                                              | 2022                     |          |                                                              |      |
| [Q-learning-based parameter control in differential evolution for structural optimization](https://www.sciencedirect.com/science/article/abs/pii/S1568494621003872) |                                                              |                                                              |                                                              | 2021                     |          |                                                              |      |
| [Reinforcement learning-based differential evolution for parameters extraction of photovoltaic models](https://www.sciencedirect.com/science/article/pii/S2352484721000974) |                                                              |                                                              |                                                              | 2021                     |          |                                                              |      |
| **2.RL辅助EA-其他**                                          | [Model-predictive control via cross-entropy and gradient-based optimization](https://proceedings.mlr.press/v120/bharadhwaj20a/bharadhwaj20a.pdf) | 模型预测控制                                                 | https://github.com/homangab/gradcem                          |                          | 2020     |                                                              |      |
| [Learning off-policy with online planning](https://arxiv.org/abs/2008.10066) |                                                              | https://github.com/hari-sikchi/LOOP                          |                                                              | 2021                     |          |                                                              |      |
| [Temporal difference learning for model predictive control](https://arxiv.org/abs/2203.04955) | 模型预测控制                                                 | https://github.com/nicklashansen/tdmpc                       |                                                              | 2022                     |          |                                                              |      |
| 3**.RL EA相辅相成-单智能体优化**                             | [EvoRainbow: Combining Improvements in Evolutionary Reinforcement Learning for Policy Search](https://openreview.net/forum?id=75Hes6Zse4) |                                                              | https://github.com/yeshenpy/EvoRainbow                       |                          | 2024     |                                                              |      |
| [Value-Evolutionary-Based Reinforcement Learning](https://openreview.net/forum?id=XobPpcN4yZ) |                                                              | https://github.com/yeshenpy/VEB-RL                           |                                                              | 2024                     |          |                                                              |      |
| [ERL-Re2: Efficient Evolutionary Reinforcement Learning with Shared State Representation and Individual Policy Representation](https://arxiv.org/abs/2210.17375) |                                                              | https://github.com/yeshenpy/ERL-Re2                          |                                                              | 2023                     |          |                                                              |      |
| [PGPS: Coupling Policy Gradient with Population-based Search](https://openreview.net/forum?id=PeT5p3ocagr) |                                                              | https://github.com/NamKim88/PGPS/blob/master/Main.py         |                                                              | 2021                     |          |                                                              |      |
| [Off-policy evolutionary reinforcement learning with maximum mutations (Maximum Mutation Reinforcement Learning for Scalable Control)](https://nbviewer.org/github/karush17/karush17.github.io/blob/master/_pages/temp4.pdf) |                                                              | https://github.com/karush17/esac                             |                                                              | 2022                     |          |                                                              |      |
| [Evolutionary action selection for gradient-based policy learning](https://arxiv.org/abs/2201.04286v1) |                                                              |                                                              |                                                              | 2022                     |          |                                                              |      |
| [Competitive and cooperative heterogeneous deep reinforcement learning](https://arxiv.org/abs/2011.00791) |                                                              |                                                              |                                                              | 2020                     |          |                                                              |      |
| [Guiding Evolutionary Strategies with Off-Policy Actor-Critic](https://dl.acm.org/doi/10.5555/3463952.3464104) |                                                              |                                                              |                                                              |                          |          |                                                              |      |
| [PDERL: Proximal Distilled Evolutionary Reinforcement Learning](https://arxiv.org/abs/1906.09807) |                                                              | https://github.com/crisbodnar/pderl                          |                                                              | 2020                     |          |                                                              |      |
| [Gradient Bias to Solve the Generalization Limit of Genetic Algorithms Through Hybridization with Reinforcement Learning](https://dl.acm.org/doi/abs/10.1007/978-3-030-64583-0_26) |                                                              | https://github.com/ricordium/Gradient-Bias                   |                                                              | 2020                     |          |                                                              |      |
| [Collaborative Evolutionary Reinforcement Learning](https://arxiv.org/abs/1905.00976) |                                                              | https://github.com/intelai/cerl                              |                                                              | 2019                     |          |                                                              |      |
| [Evolution-Guided Policy Gradient in Reinforcement Learning](https://arxiv.org/abs/1810.01222) |                                                              | https://github.com/apourchot/CEM-RL                          |                                                              | 2018                     |          |                                                              |      |
| 3**.RL EA相辅相成-多智能体优化**                             | [RACE: Improve Multi-Agent Reinforcement Learning with Representation Asymmetry and Collaborative Evolution](https://icml.cc/virtual/2023/poster/23791) |                                                              | https://github.com/yeshenpy/RACE                             |                          | 2023     |                                                              |      |
| [Novelty Seeking Multiagent Evolutionary Reinforcement Learning](https://dl.acm.org/doi/abs/10.1145/3583131.3590428) |                                                              |                                                              |                                                              | 2023                     |          |                                                              |      |
| [Evolution Strategies Enhanced Complex Multiagent Coordination](https://ieeexplore.ieee.org/document/10191313) |                                                              |                                                              |                                                              | 2023                     |          |                                                              |      |
| [MAEDyS: multiagent evolution via dynamic skill selection](https://dl.acm.org/doi/abs/10.1145/3449639.3459387) |                                                              |                                                              |                                                              | 2021                     |          |                                                              |      |
| [Evolutionary Reinforcement Learning for Sample-Efficient Multiagent Coordination](https://arxiv.org/abs/1906.07315) |                                                              | https://anonymous.4open.science/repository/1590ffb0-aa6b-4838-9d59-ae20cdd8df11/README.md https://github.com/ShawK91/MERL |                                                              | 2020                     |          |                                                              |      |
| 3**.RL EA相辅相成-形态进化**                                 | [Evolution gym: A large-scale benchmark for evolving soft robots](https://dl.acm.org/doi/abs/10.1145/3449639.3459387) |                                                              | [http://evogym.csail.mit.edu](http://evogym.csail.mit.edu/)  |                          | 2021     |                                                              |      |
| [Embodied Intelligence via Learning and Evolution](https://arxiv.org/abs/2102.02202) |                                                              | https://github.com/agrimgupta92/derl                         | Nature Communications                                        | 2021                     |          |                                                              |      |
| [Task-Agnostic Morphology Evolution](https://arxiv.org/abs/2102.13100) |                                                              | https://github.com/jhejna/morphology-opt                     |                                                              | 2021                     |          |                                                              |      |
|                                                              |                                                              |                                                              |                                                              |                          |          |                                                              |      |
|                                                              |                                                              |                                                              |                                                              |                          |          |                                                              |      |
| 3**.RL EA相辅相成-可解释AI**                                 | [Interpretable-AI Policies using Evolutionary Nonlinear Decision Trees for Discrete Action Systems](https://ieeexplore.ieee.org/document/9805655) |                                                              | https://github.com/yddhebar/NLDT                             |                          | 2024     |                                                              |      |
| [Interpretable ai for policy-making in pandemics](https://arxiv.org/abs/2204.04256) |                                                              |                                                              |                                                              | 2022                     |          |                                                              |      |
| [A co-evolutionary approach to interpretable reinforcement learning in environments with continuous action spaces](https://ieeexplore.ieee.org/document/9660048) |                                                              |                                                              |                                                              | 2021                     |          |                                                              |      |
| [Quality diversity evolutionary learning of decision trees](https://arxiv.org/abs/2204.04256) |                                                              |                                                              |                                                              | 2023                     |          |                                                              |      |
| [Social Interpretable Reinforcement Learning](https://arxiv.org/abs/2401.15480) |                                                              |                                                              |                                                              | 2024                     |          |                                                              |      |
| [Symbolic regression methods for reinforcement learning](https://arxiv.org/abs/2204.04256) |                                                              |                                                              |                                                              | 2021                     |          |                                                              |      |
| 3**.RL EA相辅相成-学习分类器系统**                           | [Classifier fitness based on accuracy](https://dl.acm.org/doi/10.1162/evco.1995.3.2.149) |                                                              | https://github.com/hosford42/xcs                             | Evolutionary computation | 1995     |                                                              |      |
| [Dynamical genetic programming in XCSF](https://pubmed.ncbi.nlm.nih.gov/22564070/) |                                                              |                                                              |                                                              | 2013                     |          |                                                              |      |
| [XCSF with tile coding in discontinuous action-value landscapes](https://link.springer.com/article/10.1007/s12065-015-0129-7) |                                                              |                                                              | Evolutionary Intelligence                                    | 2015                     |          |                                                              |      |
| [Classifiers that approximate functions](https://link.springer.com/article/10.1023/A:1016535925043) |                                                              |                                                              | Natural Computing                                            | 2002                     |          |                                                              |      |
|                                                              |                                                              |                                                              |                                                              |                          |          |                                                              |      |
|                                                              |                                                              |                                                              |                                                              |                          |          |                                                              |      |

# 团队训练

| **Paper**                                                    | **Code** | **Accepted at** | **Year** |
| ------------------------------------------------------------ | -------- | --------------- | -------- |
| [AlphaStar：Grandmaster level in StarCraft II using multi-agent reinforcement learning](https://www.gwern.net/docs/rl/2019-vinyals.pdf) |          | Nature          | 2019     |
|                                                              |          |                 |          |
|                                                              |          |                 |          |

# 课程学习

| **Paper**                                                    | **KeyWords**                                                 | **Code**                                                     | **Accepted at**                                              | **Year** |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | -------- |
| [Diverse Auto-Curriculum is Critical for Successful Real-World Multiagent Learning Systems](https://arxiv.org/abs/2102.07659) |                                                              |                                                              | AAMAS                                                        | 2021     |
| [From Few to More: Large-Scale Dynamic Multiagent Curriculum Learning](https://arxiv.org/abs/1909.02790) |                                                              | https://github.com/starry-sky6688/MARL-Algorithms            | AAAI                                                         | 2020     |
| [EPC：Evolutionary Population Curriculum for Scaling Multi-Agent Reinforcement Learning](https://arxiv.org/pdf/2003.10423) |                                                              | https://github.com/qian18long/epciclr2020                    | ICLR                                                         | 2020     |
| [Emergent Tool Use From Multi-Agent Autocurricula](https://arxiv.org/pdf/1909.07528) |                                                              | https://github.com/openai/multi-agent-emergence-environments | ICLR                                                         | 2020     |
| [Learning to Teach in Cooperative Multiagent Reinforcement Learning](https://ojs.aaai.org/index.php/AAAI/article/download/4570/4448) |                                                              |                                                              | AAAI                                                         | 2019     |
| [StarCraft Micromanagement with Reinforcement Learning and Curriculum Transfer Learning](https://arxiv.org/pdf/1804.00810) |                                                              |                                                              | IEEE Transactions on Emerging Topics in Computational Intelligence | 2018     |
| [Cooperative Multi-agent Control using deep reinforcement learning](http://ala2017.it.nuigalway.ie/papers/ALA2017_Gupta.pdf) |                                                              | https://github.com/sisl/MADRL                                | AAMAS                                                        | 2017     |
| [Variational Automatic Curriculum Learning for Sparse-Reward Cooperative Multi-Agent Problems](https://proceedings.neurips.cc/paper/2021/file/503e7dbbd6217b9a591f3322f39b5a6c-Paper.pdf) |                                                              |                                                              | NIPS                                                         | 2021     |
| [Bootstrapped Transformer for Offline Reinforcement Learning](https://arxiv.org/abs/2206.08569) | Generation model利用已学习的模型自动生成更多的离线数据，以提升序列模型的训练效果 | https://seqml.github.io/bootorl                              |                                                              |          |

# 平均场

| **Paper**                                                    | **KeyWords**                   | **Code**                                                     | **Accepted at**               | **Year** | **Others**                                                   |
| ------------------------------------------------------------ | ------------------------------ | ------------------------------------------------------------ | ----------------------------- | -------- | ------------------------------------------------------------ |
| [Mean Field Multi-Agent Reinforcement Learning](http://proceedings.mlr.press/v80/yang18d/yang18d.pdf) |                                |                                                              | ICML                          | 2018     |                                                              |
| [Efficient Ridesharing Order Dispatching with Mean Field Multi-Agent Reinforcement Learning](https://arxiv.org/pdf/1901.11454) |                                |                                                              | The world wide web conference | 2019     |                                                              |
| [Bayesian Multi-type Mean Field Multi-agent Imitation Learning](https://www.researchgate.net/profile/Wen_Dong5/publication/347240659_Bayesian_Multi-type_Mean_Field_Multi-agent_Imitation_Learning/links/5fd8c3b245851553a0bb78b1/Bayesian-Multi-type-Mean-Field-Multi-agent-Imitation-Learning.pdf) |                                |                                                              | NIPS                          | 2020     |                                                              |
| [Bridging mean-field games and normalizing flows with trajectory regularization](https://arxiv.org/pdf/2206.14990) | 联系平均场博弈与归一化流Python | https://github.com/Whalefishin/MFG_NF                        |                               | 2023     |                                                              |
|                                                              | Python                         | https://github.com/hsvgbkhgbv/Mean-field-Fictitious-Play-in-Potential-Games |                               |          |                                                              |
| [MFGLib A Library for Mean Field Games](https://arxiv.org/pdf/2304.08630) | 库                             | https://github.com/radar-research-lab/MFGLib                 |                               | 2023     |                                                              |
| [APAC-Net: Alternating the population and agent control via two neural networks to solve high-dimensional stochastic mean field games](https://arxiv.org/pdf/2002.10113) | 求解随机均场博弈100维          | https://github.com/atlin23/apac-net                          |                               | 2021     | ![img](https://e8bw0pe0za.feishu.cn/space/api/box/stream/download/asynccode/?code=NDcwZjg3OWM2MmQ1M2ExMzRmOTZjYTdkNGMxYzNlZjRfYjBpdE9ja0txVzA4RjVPaHJobFYwN2tUdk02dVVIOEFfVG9rZW46TWR3d2J3cDk2b2ZmTGV4cmR3VGNrUHhIbmxjXzE3NDU1NDU4MzA6MTc0NTU0OTQzMF9WNA) |

# 迁移学习

| **Paper**                                                    | **Code** | **Accepted at**                             | **Year** |
| ------------------------------------------------------------ | -------- | ------------------------------------------- | -------- |
| [A Survey on Transfer Learning for Multiagent Reinforcement Learning Systems](https://www.jair.org/index.php/jair/article/download/11396/26482) |          | Journal of Artificial Intelligence Research | 2019     |
| [Parallel Knowledge Transfer in Multi-Agent Reinforcement Learning](https://arxiv.org/pdf/2003.13085) |          |                                             | 2020     |
|                                                              |          |                                             |          |
|                                                              |          |                                             |          |

# 元学习

| **Paper**                                                    | **keyWords**                                                 | **Code**                          | **Accepted at** | **Year** |
| ------------------------------------------------------------ | ------------------------------------------------------------ | --------------------------------- | --------------- | -------- |
| [A Policy Gradient Algorithm for Learning to Learn in Multiagent Reinforcement Learning](https://arxiv.org/pdf/2011.00382) |                                                              |                                   | ICML            | 2021     |
| [Continuous Adaptation via Meta-Learning in Nonstationary and Competitive Environments](https://arxiv.org/pdf/1710.03641.pdf?source=post_page---------------------------) |                                                              |                                   |                 | 2017     |
| [MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework](https://arxiv.org/pdf/2308.00352) | 元编程框架：流水线范式智能体角色任务分解：子任务更连贯的解决方案 | https://github.com/geekan/MetaGPT |                 | 2023     |
|                                                              |                                                              |                                   |                 |          |

# 公平性

| **Paper**                                                    | **Code** | **Accepted at** | **Year** |
| ------------------------------------------------------------ | -------- | --------------- | -------- |
| [FEN：Learning Fairness in Multi-Agent Systems](https://arxiv.org/pdf/1910.14472) |          | NIPS            | 2019     |
| [Fairness in Multiagent Resource Allocation with Dynamic and Partial Observations](https://hal.archives-ouvertes.fr/hal-01808984/file/aamas-distrib-fairness-final.pdf) |          | AAMAS           | 2018     |
| [Fairness in Multi-agent Reinforcement Learning for Stock Trading](https://arxiv.org/pdf/2001.00918) |          |                 | 2019     |
|                                                              |          |                 |          |

# 奖励搜索

## 稠密奖励搜索

| **Paper**                                                    | **Code**                                          | **Accepted at** | **Year** |
| ------------------------------------------------------------ | ------------------------------------------------- | --------------- | -------- |
| [MAVEN：Multi-Agent Variational Exploration](https://arxiv.org/pdf/1910.07483) | https://github.com/starry-sky6688/MARL-Algorithms | NIPS            | 2019     |
| [Social Influence as Intrinsic Motivation for Multi-Agent Deep Reinforcement Learning](http://proceedings.mlr.press/v97/jaques19a/jaques19a.pdf) |                                                   | ICML            | 2019     |
| [Episodic Multi-agent Reinforcement Learning with Curiosity-driven Exploration](https://proceedings.neurips.cc/paper/2021/file/1e8ca836c962598551882e689265c1c5-Paper.pdf) |                                                   | NIPS            | 2021     |
| [Celebrating Diversity in Shared Multi-Agent Reinforcement Learning](https://proceedings.neurips.cc/paper/2021/file/20aee3a5f4643755a79ee5f6a73050ac-Paper.pdf) | https://github.com/lich14/CDS                     | NIPS            | 2021     |

## 稀疏奖励搜索

| **Paper**                                                    | **Code**                                 | **Accepted at** | **Year** |
| ------------------------------------------------------------ | ---------------------------------------- | --------------- | -------- |
| [EITI/EDTI：Influence-Based Multi-Agent Exploration](https://arxiv.org/pdf/1910.05512) | https://github.com/TonghanWang/EITI-EDTI | ICLR            | 2020     |
| [Cooperative Exploration for Multi-Agent Deep Reinforcement Learning](http://proceedings.mlr.press/v139/liu21j/liu21j.pdf) |                                          | ICML            | 2021     |
| [Centralized Model and Exploration Policy for Multi-Agent](https://arxiv.org/pdf/2107.06434) |                                          |                 | 2021     |
| [REMAX: Relational Representation for Multi-Agent Exploration](https://dl.acm.org/doi/abs/10.5555/3535850.3535977) |                                          | AAMAS           | 2022     |

## 未分

| **Paper**                                                    | **Code**                       | **Accepted at** | **Year** |
| ------------------------------------------------------------ | ------------------------------ | --------------- | -------- |
| [CM3: Cooperative Multi-goal Multi-stage Multi-agent Reinforcement Learning](https://arxiv.org/pdf/1809.05188) |                                | ICLR            | 2020     |
| [Coordinated Exploration via Intrinsic Rewards for Multi-Agent Reinforcement Learning](https://arxiv.org/pdf/1905.12127) |                                |                 | 2019     |
| [Exploration by Maximizing Renyi Entropy for Reward-Free RL Framework](https://arxiv.org/abs/2006.06193v3) |                                | AAAI            | 2021     |
| [Exploration-Exploitation in Multi-Agent Learning: Catastrophe Theory Meets Game Theory](https://arxiv.org/abs/2012.03083v2) |                                | AAAI            | 2021     |
| [LIIR: Learning Individual Intrinsic Reward in Multi-Agent Reinforcement Learning](http://papers.neurips.cc/paper/8691-liir-learning-individual-intrinsic-reward-in-multi-agent-reinforcement-learning.pdf) | https://github.com/yalidu/liir | NIPS            | 2019     |

# 稀疏奖励

| **Paper**                                                    | **KeyWords**               | **Code**                                                     | **Accepted at** | **Year** |      |
| ------------------------------------------------------------ | -------------------------- | ------------------------------------------------------------ | --------------- | -------- | ---- |
| [Variational Automatic Curriculum Learning for Sparse-Reward Cooperative Multi-Agent Problems](https://proceedings.neurips.cc/paper/2021/file/503e7dbbd6217b9a591f3322f39b5a6c-Paper.pdf) |                            |                                                              | NIPS            | 2021     |      |
| [Individual Reward Assisted Multi-Agent Reinforcement Learning](https://proceedings.mlr.press/v162/wang22ao/wang22ao.pdf) |                            | https://github.com/MDrW/ICML2022-IRAT                        | ICML            | 2022     |      |
| [Switch Trajectory Transformer with Distributional Value Approximation for Multi-Task Reinforcement Learning](https://arxiv.org/abs/2203.07413) | Multi-Task RLSparse Reward | ExpEnv: [MINIGRID](https://github.com/Farama-Foundation/gym-minigrid) |                 |          |      |

# 图神经网络

| **Paper**                                                    |      | **Code**                                          | **Accepted at** | **Year** |                                                              |
| ------------------------------------------------------------ | ---- | ------------------------------------------------- | --------------- | -------- | ------------------------------------------------------------ |
| [Multi-Agent Game Abstraction via Graph Attention Neural Network](https://ojs.aaai.org/index.php/AAAI/article/view/6211/6067) |      | https://github.com/starry-sky6688/MARL-Algorithms | AAAI            | 2020     |                                                              |
| [Graph Convolutional Reinforcement Learning for Multi-Agent Cooperation](https://arxiv.org/abs/1810.09202v1) |      |                                                   | ICLR            | 2020     |                                                              |
| [Multi-Agent Reinforcement Learning with Graph Clustering](https://arxiv.org/pdf/2008.08808) |      |                                                   |                 | 2020     |                                                              |
| [Learning to Coordinate with Coordination Graphs in Repeated Single-Stage Multi-Agent Decision Problems](http://proceedings.mlr.press/v80/bargiacchi18a/bargiacchi18a.pdf) |      |                                                   | ICML            | 2018     |                                                              |
| [Distributed constrained combinatorial optimization leveraging hypergraph neural networks](https://arxiv.org/pdf/2311.09375) |      | https://github.com/nasheydari/HypOp               |                 | 2023     |                                                              |
| [Learning Scalable Policies over Graphs for Multi-Robot Task Allocation using Capsule Attention Networks](https://ieeexplore.ieee.org/abstract/document/9812370/) |      | https://github.com/iamstevepaul/MRTA-Graph_RL     |                 |          | ![img](https://e8bw0pe0za.feishu.cn/space/api/box/stream/download/asynccode/?code=ZjUxYzQ2MzNjODFkMzU3ODM1NmNmYzc1MzJlNzkwM2VfSUVPYThMNGJmSlZFU3BKQWNyVzFob0RVZWZMZmp0dmZfVG9rZW46S1VqTGJucEN1b01YNDF4UDNzdGNkZjlEbnNlXzE3NDU1NDU4MzA6MTc0NTU0OTQzMF9WNA) |

# 基于模型的

| **Paper**                                                    | **Code** | **Accepted at** | **Year** |
| ------------------------------------------------------------ | -------- | --------------- | -------- |
| [Model-based Multi-Agent Reinforcement Learning with Cooperative Prioritized Sweeping](https://arxiv.org/pdf/2001.07527) |          |                 | 2020     |
|                                                              |          |                 |          |
|                                                              |          |                 |          |

# 神经架构搜索NAS

| **Paper**                                                    | **Code** | **Accepted at** | **Year** |
| ------------------------------------------------------------ | -------- | --------------- | -------- |
| [MANAS: Multi-Agent Neural Architecture Search](https://arxiv.org/pdf/1909.01051) |          |                 | 2019     |
|                                                              |          |                 |          |
|                                                              |          |                 |          |

# 安全学习

| **Paper**                                                    | **Code** | **Accepted at** | **Year** |
| ------------------------------------------------------------ | -------- | --------------- | -------- |
| [MAMPS: Safe Multi-Agent Reinforcement Learning via Model Predictive Shielding](https://arxiv.org/pdf/1910.12639) |          |                 | 2019     |
| [Safer Deep RL with Shallow MCTS: A Case Study in Pommerman](https://arxiv.org/pdf/1904.05759) |          |                 | 2019     |
|                                                              |          |                 |          |

# 单智能体到多智能体

| **Paper**                                                    | **Code**                                   | **Accepted at** | **Year** |
| ------------------------------------------------------------ | ------------------------------------------ | --------------- | -------- |
| [IQL：Multi-Agent Reinforcement Learning: Independent vs. Cooperative Agents](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.84.3701&rep=rep1&type=pdf) | https://github.com/oxwhirl/pymarl          | ICML            | 1993     |
| [IPPO：Is Independent Learning All You Need in the StarCraft Multi-Agent Challenge?](https://arxiv.org/pdf/2011.09533) |                                            |                 | 2020     |
| [MAPPO：The Surprising Effectiveness of MAPPO in Cooperative, Multi-Agent Games](https://arxiv.org/pdf/2103.01955) | https://github.com/marlbenchmark/on-policy |                 | 2021     |
| [MADDPG：Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/pdf/1706.02275.pdf">Multi-Agent) | https://github.com/openai/maddpg           | NIPS            | 2017     |

# 动作空间

| **Paper**                                                    | **Code**                                                     | **Accepted at** | **Year** |      |      |      |
| ------------------------------------------------------------ | ------------------------------------------------------------ | --------------- | -------- | ---- | ---- | ---- |
| [Deep Reinforcement Learning in Parameterized Action Space](https://arxiv.org/pdf/1511.04143) |                                                              |                 | 2015     |      |      |      |
| [DMAPQN: Deep Multi-Agent Reinforcement Learning with Discrete-Continuous Hybrid Action Spaces](https://arxiv.org/pdf/1903.04959) |                                                              | IJCAI           | 2019     |      |      |      |
| [H-PPO: Hybrid actor-critic reinforcement learning in parameterized action space](https://arxiv.org/pdf/1903.01344) |                                                              | IJCAI           | 2019     |      |      |      |
| [P-DQN: Parametrized Deep Q-Networks Learning: Reinforcement Learning with Discrete-Continuous Hybrid Action Space](https://arxiv.org/pdf/1810.06394) |                                                              |                 | 2018     |      |      |      |
| [Few is More: Task-Efficient Skill-Discovery for Multi-Task Offline Multi-Agent Reinforcement Learning](https://arxiv.org/pdf/2502.08985) | 避免新任务重复训练多任务离线MARL算法重构观测->评估固定动作+可变动作->正则保守动作从有限小规模源任务->强大的多任务泛化 |                 |          | 2025 |      |      |

# 多样性

| **Paper**                                                    | **KeyWords**                                                 | **Code**                            | **Accepted at** | **Year** |      |      |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ----------------------------------- | --------------- | -------- | ---- | ---- |
| [Diverse Auto-Curriculum is Critical for Successful Real-World Multiagent Learning Systems](https://arxiv.org/pdf/2102.07659) |                                                              |                                     | AAMAS           | 2021     |      |      |
| [Q-DPP：Multi-Agent Determinantal Q-Learning](http://proceedings.mlr.press/v119/yang20i/yang20i.pdf) |                                                              | https://github.com/QDPP-GitHub/QDPP | ICML            | 2020     |      |      |
| [Diversity is All You Need: Learning Skills without a Reward Function](https://arxiv.org/pdf/1802.06070) |                                                              |                                     |                 | 2018     |      |      |
| [Modelling Behavioural Diversity for Learning in Open-Ended Games](https://arxiv.org/pdf/2103.07927) |                                                              |                                     | ICML            | 2021     |      |      |
| [Diverse Agents for Ad-Hoc Cooperation in Hanabi](https://arxiv.org/pdf/1907.03840) |                                                              |                                     | CoG             | 2019     |      |      |
| [Generating Behavior-Diverse Game AIs with Evolutionary Multi-Objective Deep Reinforcement Learning](https://nos.netease.com/mg-file/mg/neteasegamecampus/art_works/20200812/202008122020238603.pdf) |                                                              |                                     | IJCAI           | 2020     |      |      |
| [Quantifying environment and population diversity in multi-agent reinforcement learning](https://arxiv.org/pdf/2102.08370) |                                                              |                                     |                 | 2021     |      |      |
| [POMO: Policy Optimization with Multiple Optima for Reinforcement Learning](https://arxiv.org/pdf/2010.16011) | REINFORCE算法组合优化问题多样化轨迹Python                    | https://github.com/yd-kwon/POMO     |                 | 2021     |      |      |
| [HIQL: Offline Goal-Conditioned RL with Latent States as Actions](https://arxiv.org/abs/2303.03982) | Hierarchical Goal-Conditioned RLOffline Reinforcement LearningValue Function Estimation |                                     |                 | 2023     |      |      |

# 分布式训练分布式执行

| **Paper**                                                    | **Code** | **Accepted at**                         | **Year** |
| ------------------------------------------------------------ | -------- | --------------------------------------- | -------- |
| [Networked Multi-Agent Reinforcement Learning in Continuous Spaces](https://ieeexplore.ieee.org/abstract/document/8619581) |          | IEEE conference on decision and control | 2018     |
| [Value Propagation for Decentralized Networked Deep Multi-agent Reinforcement Learning](https://proceedings.neurips.cc/paper/2019/file/8a0e1141fd37fa5b98d5bb769ba1a7cc-Paper.pdf) |          | NIPS                                    | 2019     |
| [Fully Decentralized Multi-Agent Reinforcement Learning with Networked Agents](http://proceedings.mlr.press/v80/zhang18n/zhang18n.pdf) |          | ICML                                    | 2018     |
|                                                              |          |                                         |          |

# 离线多智能体强化学习

| **Paper**                                                    | **Key Words**                                                | **Code**                                    | **Accepted at** | **Year** |      |      |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------- | --------------- | -------- | ---- | ---- |
| [Offline Pre-trained Multi-Agent Decision Transformer: One Big Sequence Model Conquers All StarCraftII Tasks](https://arxiv.org/pdf/2112.02845) |                                                              |                                             |                 | 2021     |      |      |
| [Believe what you see: Implicit constraint approach for offline multi-agent reinforcement learning](https://proceedings.neurips.cc/paper/2021/file/550a141f12de6341fba65b0ad0433500-Paper.pdf) |                                                              |                                             | NIPS            | 2021     |      |      |
| [FOCAL: Efficient Fully-Offline Meta-Reinforcement Learning via Distance Metric Learning and Behavior Regularization](https://arxiv.org/pdf/2010.01112) |                                                              | https://github.com/LanqingLi1993/FOCAL-ICLR |                 | 2020     |      |      |
| [ComaDICE: Offline Cooperative Multi-Agent Reinforcement Learning with Stationary Distribution Shift Regularization](https://arxiv.org/pdf/2410.01954) |                                                              | 分布偏移                                    |                 |          |      |      |
| [Discovering Generalizable Multi-agent Coordination Skills from Multi-task Offline Data](https://openreview.net/pdf?id=53FyUAdP7d) | 有限来源的离线数据 MARL跨任务的协作未见任务泛化能力          | https://github.com/LAMDA-RL/ODIS            |                 | 2023     |      |      |
| [Few is More: Task-Efficient Skill-Discovery for Multi-Task Offline Multi-Agent Reinforcement Learning](https://arxiv.org/pdf/2502.08985) | 避免新任务重复训练多任务离线MARL算法重构观测->评估固定动作+可变动作->正则保守动作从有限小规模源任务->强大的多任务泛化 |                                             |                 | 2025     |      |      |

# 对抗

## 单智能体

| **Paper**                                                    | **Code**                                                  | **Accepted at** | **Year** |
| ------------------------------------------------------------ | --------------------------------------------------------- | --------------- | -------- |
| [Robust Adversarial Reinforcement Learning](http://proceedings.mlr.press/v70/pinto17a/pinto17a.pdf) | Non-official implements on GitHub                         | ICML            | 2017     |
| [Robust Deep Reinforcement Learning against Adversarial Perturbations on State Observations](https://proceedings.neurips.cc/paper/2020/file/f0eb6568ea114ba6e293f903c34d7488-Paper.pdf) | https://github.com/chenhongge/StateAdvDRL                 | NIPS            | 2020     |
| [Robust Reinforcement Learning as a Stackelberg Game via Adaptively-Regularized Adversarial Training](https://arxiv.org/pdf/2202.09514) |                                                           |                 | 2022     |
| [Risk Averse Robust Adversarial Reinforcement Learning](https://arxiv.org/pdf/1904.00511) |                                                           | ICRA            | 2019     |
| [Robust Deep Reinforcement Learning with Adversarial Attacks](https://arxiv.org/pdf/1712.03632) |                                                           |                 | 2017     |
| [Robust Reinforcement Learning on State Observations with Learned Optimal Adversary](https://arxiv.org/pdf/2101.08452) | https://github.com/huanzhang12/ATLA_robust_RL             | ICLR            | 2021     |
| [Exploring the Training Robustness of Distributional Reinforcement Learning against Noisy State Observations](https://arxiv.org/pdf/2109.08776) |                                                           |                 | 2021     |
| [RoMFAC: A Robust Mean-Field Actor-Critic Reinforcement Learning against Adversarial Perturbations on States](https://arxiv.org/pdf/2205.07229) |                                                           |                 | 2022     |
| [Adversary Agnostic Robust Deep Reinforcement Learning](https://arxiv.org/pdf/2008.06199) |                                                           | TNNLS           | 2021     |
| [Learning to Cope with Adversarial Attacks](https://arxiv.org/pdf/1906.12061) |                                                           |                 | 2019     |
| [Adversarial Attack on Graph Structured Data](http://proceedings.mlr.press/v80/dai18b/dai18b.pdf) |                                                           | ICML            | 2018     |
| [Characterizing Attacks on Deep Reinforcement Learning](http://proceedings.mlr.press/v80/dai18b/dai18b.pdf) |                                                           | AAMAS           | 2022     |
| [Adversarial policies: Attacking deep reinforcement learning](https://arxiv.org/pdf/1905.10615) | https://github.com/HumanCompatibleAI/adversarial-policies | ICLR            | 2020     |
| [Learning Robust Policy against Disturbance in Transition Dynamics via State-Conservative Policy Optimization](https://ojs.aaai.org/index.php/AAAI/article/view/20686/20445) |                                                           | AAAI            | 2022     |
| [On the Robustness of Safe Reinforcement Learning under Observational Perturbations](https://arxiv.org/pdf/2205.14691) |                                                           |                 | 2022     |
| [Robust Reinforcement Learning using Adversarial Populations](https://arxiv.org/pdf/2008.01825) |                                                           |                 | 2020     |
| [Robust Deep Reinforcement Learning through Adversarial Loss](https://proceedings.neurips.cc/paper/2021/file/dbb422937d7ff56e049d61da730b3e11-Paper.pdf) | https://github.com/tuomaso/radial_rl_v2                   | NIPS            | 2021     |

## 多智能体

| **Paper**                                                    | **Code** | **Accepted at**                           | **Year** |
| ------------------------------------------------------------ | -------- | ----------------------------------------- | -------- |
| [Certifiably Robust Policy Learning against Adversarial Communication in Multi-agent Systems](https://arxiv.org/pdf/2206.10158) |          |                                           | 2022     |
| [Distributed Multi-Agent Deep Reinforcement Learning for Robust Coordination against Noise](https://arxiv.org/pdf/2205.09705) |          |                                           | 2022     |
| [On the Robustness of Cooperative Multi-Agent Reinforcement Learning](https://arxiv.org/pdf/2003.03722) |          | IEEE Security and Privacy Workshops       | 2020     |
| [Towards Comprehensive Testing on the Robustness of Cooperative Multi-agent Reinforcement Learning](https://openaccess.thecvf.com/content/CVPR2022W/ArtOfRobust/papers/Guo_Towards_Comprehensive_Testing_on_the_Robustness_of_Cooperative_Multi-Agent_Reinforcement_CVPRW_2022_paper.pdf) |          | CVPR workshop                             | 2022     |
| [Robust Multi-Agent Reinforcement Learning via Minimax Deep Deterministic Policy Gradient](https://ojs.aaai.org/index.php/AAAI/article/view/4327/4205) |          | AAAI                                      | 2019     |
| [Multi-agent Deep Reinforcement Learning with Extremely Noisy Observations](https://arxiv.org/pdf/1812.00922) |          | NIPS Deep Reinforcement Learning Workshop | 2018     |
| [Policy Regularization via Noisy Advantage Values for Cooperative Multi-agent Actor-Critic methods](https://arxiv.org/pdf/2106.14334) |          |                                           | 2021     |

## 对抗通信

| **Paper**                                                    | **Code** | **Accepted at** | **Year** |
| ------------------------------------------------------------ | -------- | --------------- | -------- |
| [Certifiably Robust Policy Learning against Adversarial Communication in Multi-agent Systems](https://arxiv.org/pdf/2206.10158) |          |                 | 2022     |
|                                                              |          |                 |          |
|                                                              |          |                 |          |

## 评估

| **Paper**                                                    | **Code** | **Accepted at** | **Year** |
| ------------------------------------------------------------ | -------- | --------------- | -------- |
| [Towards Comprehensive Testing on the Robustness of Cooperative Multi-agent Reinforcement Learning](https://openaccess.thecvf.com/content/CVPR2022W/ArtOfRobust/papers/Guo_Towards_Comprehensive_Testing_on_the_Robustness_of_Cooperative_Multi-Agent_Reinforcement_CVPRW_2022_paper.pdf) |          | CVPR workshop   | 2022     |
|                                                              |          |                 |          |
|                                                              |          |                 |          |

# 模仿学习

| **Paper**                                                    | **Key Words**                                   | **Code**                                       | **Accepted at** | **Year** | **Others**                                                   |      |
| ------------------------------------------------------------ | ----------------------------------------------- | ---------------------------------------------- | --------------- | -------- | ------------------------------------------------------------ | ---- |
| [MAPF-GPT: Imitation Learning for Multi-Agent Pathfinding at Scale](https://arxiv.org/abs/2409.00134) | 路径发现无通信 无启发式模仿学习部分可观测Python | https://github.com/CognitiveAISystems/MAPF-GPT |                 | 2025     | ![img](https://e8bw0pe0za.feishu.cn/space/api/box/stream/download/asynccode/?code=OGIyZTZhNjk4ZTRkNWJjODhjNTQ0MmM4MmIxZmUwZmVfcXdEZXFYUDdtWkZ3enc1MXBOR2lTbXRCQ1VwRzZld2dfVG9rZW46TGt3YmJ0NVlYb1hqSHB4SXdCS2NXdThRbmplXzE3NDU1NDU4MzA6MTc0NTU0OTQzMF9WNA) |      |
|                                                              |                                                 |                                                |                 |          |                                                              |      |
|                                                              |                                                 |                                                |                 |          |                                                              |      |

# 训练数据

| **Paper**                                                    | **Key Words**                                                | **Code**                         | **Accepted at** | **Year** |      |      |
| ------------------------------------------------------------ | ------------------------------------------------------------ | -------------------------------- | --------------- | -------- | ---- | ---- |
| [INS: Interaction-aware Synthesis to Enhance Offline Multi-agent Reinforcement Learning](https://openreview.net/pdf?id=kxD2LlPr40) | 数据稀缺性智能体间交互数据扩散模型合成高质量多智能体数据集稀疏注意力机制 | https://github.com/fyqqyf/INS    |                 | 2025     |      |      |
| [Discovering Generalizable Multi-agent Coordination Skills from Multi-task Offline Data](https://openreview.net/pdf?id=53FyUAdP7d) | 有限来源的离线数据 MARL跨任务的协作未见任务泛化能力          | https://github.com/LAMDA-RL/ODIS |                 | 2023     |      |      |
| [Few is More: Task-Efficient Skill-Discovery for Multi-Task Offline Multi-Agent Reinforcement Learning](https://arxiv.org/pdf/2502.08985) | 避免新任务重复训练多任务离线MARL算法重构观测->评估固定动作+可变动作->正则保守动作从有限小规模源任务->强大的多任务泛化 |                                  |                 | 2025     |      |      |

# 优化器

| **Paper**                                                    | **Key Words**                                                | **Code**                        | **Accepted at** | **Year** |      |      |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------- | --------------- | -------- | ---- | ---- |
| [Conformal Symplectic Optimization for Stable Reinforcement Learning](https://arxiv.org/pdf/2412.02291) | RL专用神经网络优化器RAD性能达到Adam优化器的2.5倍得分提升了155.1% | https://github.com/TobiasLv/RAD |                 | 2025     |      |      |
|                                                              |                                                              |                                 |                 |          |      |      |
|                                                              |                                                              |                                 |                 |          |      |      |

# 待分类

| **Paper**                                                    | **KeyWords**                                    | **Code**                                                     | **Accepted at** | **Year** |
| ------------------------------------------------------------ | ----------------------------------------------- | ------------------------------------------------------------ | --------------- | -------- |
| [Mind-aware Multi-agent Management Reinforcement Learning](https://arxiv.org/pdf/1810.00147) |                                                 | https://github.com/facebookresearch/M3RL                     | ICLR            | 2019     |
| [Emergence of grounded compositional language in multi-agent populations](https://ojs.aaai.org/index.php/AAAI/article/download/11492/11351) |                                                 | https://github.com/bkgoksel/emergent-language                | AAAI            | 2018     |
| [Emergent Complexity via Multi-Agent Competition](https://arxiv.org/pdf/1710.03748.pdfKEYWORDS: Artificial) |                                                 | https://github.com/openai/multiagent-competition             | ICLR            | 2018     |
| [TLeague: A Framework for Competitive Self-Play based Distributed Multi-Agent Reinforcement Learning](https://arxiv.org/pdf/2011.12895) |                                                 | https://github.com/tencent-ailab/TLeague                     |                 | 2020     |
| [UPDeT: Universal Multi-agent Reinforcement Learning via Policy Decoupling with Transformers](https://openreview.net/forum?id=v9c7hr9ADKx) |                                                 | https://github.com/hhhusiyi-monash/UPDeT                     | ICLR            | 2021     |
| [SIDE: State Inference for Partially Observable Cooperative Multi-Agent Reinforcement Learning](https://www.ifaamas.org/Proceedings/aamas2022/pdfs/p1400.pdf) |                                                 | https://github.com/deligentfool/SIDE                         | AAMAS           | 2022     |
| [UNMAS: Multiagent Reinforcement Learningfor Unshaped Cooperative Scenarios](https://arxiv.org/pdf/2203.14477) |                                                 | https://github.com/James0618/unmas                           | TNNLS           | 2021     |
| [Context-Aware Sparse Deep Coordination Graphs](https://arxiv.org/pdf/2106.02886) |                                                 | https://github.com/TonghanWang/CASEC-MACO-benchmark          | ICLR            | 2022     |
| [Neural Spline Flows](https://arxiv.org/pdf/1906.04032)      | 流模型概率密度评估和采样模型灵活性PyTorchPython | https://github.com/bayesiains/nflowshttps://github.com/bayesiains/nsf |                 | 2021     |
| 领域知识图谱数据采集、数据处理以及可视化                     |                                                 | https://github.com/Louis-tiany/Military-KG                   |                 |          |
|                                                              |                                                 |                                                              |                 |          |

# 致谢
[Chen, Hao, Multi-Agent Reinforcement Learning Papers with Code](https://github.com/TimeBreaker/MARL-papers-with-code)

[Chen, Hao, Multi Agent Reinforcement Learning papers](https://github.com/TimeBreaker/Multi-Agent-Reinforcement-Learning-papers)

[Chen, Hao, MARL Resources Collection](https://github.com/TimeBreaker/MARL-resources-collection)

