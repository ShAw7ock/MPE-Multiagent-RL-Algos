MPE-Multiagent-Reinforcement Learning-Algorithms
=======================
## MPE
This is a simple verification experiments codes for `Multi-Agent RL` using OpenAI [Multi-agent Particle Environment](https://github.com/openai/multiagent-particle-envs).<br>
The environment includes many benchmarks and originally prepare for [MADDPG](https://arxiv.org/pdf/1706.02275.pdf) algorithm.<br>
Since the simple environment settings, `MPE` seems to be a good toy experienments' environment to verify our new Multi-Agent RL algorithms and compare with other baselines.<br>
I write down some famous multi-agent RL algorithms for you so that you could change fewer codes to realize your own algorithms and verify the experiments results.<br>
<br>

<p align="center">
 Simple Spread Demo<br>
 <img src="https://github.com/ShAw7ock/MPE-Multiagent-RL-Algos/blob/master/models/simple_spread/vdn/run3/results/VDN_Simple_Spread.gif" width="352" height="352"><br>
</p>

<p align="center">
 Algorithm Comparing<br>
 <img src="https://github.com/ShAw7ock/MPE-Multiagent-RL-Algos/blob/master/models/myplot.png" width="640" height="480">
</p>

***NOTE:*** <br>
* We assume that the default scenario `"simple_spread"` (complete cooperation setting) is the ***ONLY*** fitted one.<br>
* And I have modified the MPE environment to fit the QMIX and LIIR algorithms (add the total state). So I suggest you to create a new virtual environment (Anaconda ...) to download my MPE repo.
* The [OpenAI Baselines](https://github.com/openai/baselines) is REQUIRED for multi-process environment, and you can also use the baselines files in this repo.
## Requirements
* Python >= 3.6.0
* PyTorch == 1.2.0
* OpenAI Gym == 0.10.5
* [Multi-agent Particle Environment (MPE)](https://github.com/openai/multiagent-particle-envs)
## Algorithms
- [x] [VDN](https://arxiv.org/pdf/1706.05296.pdf)
- [x] [QMIX](http://proceedings.mlr.press/v80/rashid18a/rashid18a.pdf)
- [x] [COMA](https://ojs.aaai.org/index.php/AAAI/article/view/11794)
- [x] [LIIR](https://proceedings.neurips.cc/paper/2019/file/07a9d3fed4c5ea6b17e80258dee231fa-Paper.pdf)
- [x] [MAAC](http://proceedings.mlr.press/v97/iqbal19a/iqbal19a.pdf)
## Running
- To run the code, `cd` into the root directory and run `main.py`:
``python main.py --algo ALGO_NAME``
- You can replace the `ALGO_NAME` with the algorithms I have mentioned (`vdn`,`qmix`,`coma`,`liir` and `maac`).
- To evaluate your saved model, `cd` into the root directory and run `evaluate.py`:
``python evaluate.py --algo ALGO_NAME --run_num NUMBER``
- You can replace the `NUMBER` with the model-saved file num, for example, you have saved the VDN model in `./models/simple_spread/vdn/run2` and you wanna evaluate it, then you should replace NUMBER with 2.
## TODO List
- [x] Evaluate and rendering
- [x] Figures and comparing
- [x] Upload the training models.pt
- [ ] Multi-threading with creating envs

Acknowledgement
---------------
* I have been studying as a master student. There may be some problems with my codes and understanding of the algorithms.<br>
* `Shariq Iqbal`'s [MADDPG-PyTorch Codes](https://github.com/shariqiqbal2810/maddpg-pytorch) and `starry-sky6688`'s [StarCraft Multi-Agent RL Codes](https://github.com/starry-sky6688/StarCraft) are used as references.Of course, OpenAI opening their codes of `MADDPG` and `Multi-agent Particle Environment` also gives a lot of help.Thanks for their contributions to the open source world.<br>

***Thanks for using `ShAw7ock`'s codes.***
