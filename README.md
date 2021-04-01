MPE-Multiagent-Reinforcement Learning-Algorithms
=======================
## MPE
This is a simple verification experiments codes for `Multi-Agent RL` using OpenAI [Multi-agent Particle Environment](https://github.com/openai/multiagent-particle-envs).<br>
The environment concludes many benchmarks and originally prepare for [MADDPG](https://arxiv.org/pdf/1706.02275.pdf) algorithm.<br>
Since the simple environment settings, `MPE` seems to be a good toy experienments' environment to verify our new Multi-Agent RL algorithms and compare with other baselines.<br>
I write down some famous multi-agent RL algorithms for you so that you could change fewer codes to realize your own algorithms and verify the experiments results.<br>
NOTE: <br>
* If you wanna run this MPE environment successfully, you have to make sure you have download the [OpenAI Baselines](https://github.com/openai/baselines).<br>
* However, I have push the baselines' files into the project, if you find anything wrong with the baselines you download from OpenAI, you could just use the files of mine.<br>
## Requirements
* Python >= 3.6.0
* PyTorch == 1.2.0
* OpenAI Gym == 0.10.5
* [Multi-agent Particle Environment](https://github.com/openai/multiagent-particle-envs)
## Algorithms
- [x] [VDN](https://arxiv.org/pdf/1706.05296.pdf)
- [ ] [QMIX](http://proceedings.mlr.press/v80/rashid18a/rashid18a.pdf)
- [ ] [COMA](https://ojs.aaai.org/index.php/AAAI/article/view/11794)
- [ ] [LIIR](https://proceedings.neurips.cc/paper/2019/file/07a9d3fed4c5ea6b17e80258dee231fa-Paper.pdf)
## TODO List
- [ ] Evaluate and rendering
- [ ] Figures and comparing
- [ ] Upload the training models.pt

Acknowledgement
---------------
I have been studying as a master student. There may be some problems with my codes and understanding of the algorithms.<br>
Thanks for using `ShAw7ock`'s codes.
