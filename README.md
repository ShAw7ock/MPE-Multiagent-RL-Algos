MPE-Multiagent-Reinforcement Learning-Algorithms
=======================
## MPE
This is a simple verification experiments codes for `Multi-Agent RL` using OpenAI [Multi-agent Particle Environment](https://github.com/openai/multiagent-particle-envs).<br>
The environment concludes many benchmarks and originally prepare for [MADDPG](https://arxiv.org/pdf/1706.02275.pdf) algorithm.<br>
Since the simple environment settings, `MPE` seems to be a good toy experienments' environment to verify our new Multi-Agent RL algorithms and compare with other baselines.<br>
I write down some famous multi-agent RL algorithms for you so that you could change fewer codes to realize your own algorithms and verify the experiments results.<br>
<br>
The codes can only be used with the one of the benchmarks named `"simple_spread"` (env_id) which is a complete cooperation setting.<br>
If the codes would like to adapt the other envs like MADDPG codes did, there would be much other work to do.<br>
However, our goal to use the simple MPE environment is to verify our new algorithm, the `"simple_spread"` is enough. The next experiments which could be put into the papers may be the [StarCraft-SMAC](https://github.com/oxwhirl/smac) or something else.<br>
<br>
***NOTE:*** <br>
* If you wanna run this MPE environment successfully, you have to make sure you have download the [OpenAI Baselines](https://github.com/openai/baselines).<br>
* However, I have push the baselines' files into the project, if you find anything wrong with the baselines you download from OpenAI, you could just use the files of mine.<br>
## Requirements
* Python >= 3.6.0
* PyTorch == 1.2.0
* OpenAI Gym == 0.10.5
* [Multi-agent Particle Environment (MPE)](https://github.com/openai/multiagent-particle-envs)
## Algorithms
- [x] [VDN](https://arxiv.org/pdf/1706.05296.pdf)
- [x] [QMIX](http://proceedings.mlr.press/v80/rashid18a/rashid18a.pdf)
- [x] [COMA](https://ojs.aaai.org/index.php/AAAI/article/view/11794)
- [ ] [LIIR](https://proceedings.neurips.cc/paper/2019/file/07a9d3fed4c5ea6b17e80258dee231fa-Paper.pdf)
## TODO List
- [ ] Evaluate and rendering
- [ ] Figures and comparing
- [ ] Upload the training models.pt
- [ ] Multi-threading with creating envs

Acknowledgement
---------------
I have been studying as a master student. There may be some problems with my codes and understanding of the algorithms.<br>
`Shariq Iqbal`'s [MADDPG-PyTorch Codes](https://github.com/shariqiqbal2810/maddpg-pytorch) and `starry-sky6688`'s [StarCraft Multi-Agent RL Codes](https://github.com/starry-sky6688/StarCraft) are used as references.Of course, OpenAI opening their codes of `MADDPG` and `Multi-agent Particle Environment` also gives a lot of help.Thanks for their contributions to the open source world.<br>
***Thanks for using `ShAw7ock`'s codes.***