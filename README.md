# Human-Level Control without Server-Grade Hardware

[Deep Q-Network (DQN)](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf)
was a landmark achievement for reinforcement learning (RL) by generating human-level policies for playing Atari games directly from pixels and a reward signal.
Although first published back in 2015, DQN still requires an enormous amount of computation to fully replicate the original results on all 49 games.

This code provides a modified DQN implementation whose speed is optimized for multi-core, single-GPU computers.
The goal of this project is to promote access to deep RL by providing a fast and well-tested DQN baseline that does not require large server clusters to be feasible.

For more details, see the accompanying paper:
[Human-Level Control without Server-Grade Hardware](https://arxiv.org/abs/2111.01264).
To cite this code in published work, you can just cite the paper itself:

**Citation**

```
@article{daley2021human,
  title={Human-Level Control without Server-Grade Hardware},
  author={Daley, Brett and Amato, Christopher},
  journal={arXiv preprint arXiv:2106.05449},
  year={2021}
}
```


## Setup

The code is based on Python 3.5+ and TensorFlow 2.

To get started, clone this repository and install the required packages:

```
git clone https://github.com/brett-daley/fast-dqn.git
cd fast-dqn
pip install -r requirements.txt
```

Make sure that appropriate versions of CUDA and cuDNN are also installed to enable
[TensorFlow with GPU support](https://www.tensorflow.org/install/gpu).


## Usage

### Fast DQN

To train our fast DQN implementation on the game Pong, execute the following command:

`python run_fast_dqn.py --game=pong --workers=8`

This example dispatches 8 samplers ("workers") each in their own game instance, with both Concurrent Training and Synchronized Execution enabled by default
(see [How It Works](#how-it-works)).
Generally speaking, the number of samplers should match the number of threads available on your CPU for the best performance.

For other possible arguments, refer to `python run_fast_dqn.py --help`.


### Supported Environments

Currently, the code has some hardcoded assumptions that restrict it to playing Atari games only.
In particular, the `--game` argument assumes that it will receive a string that corresponds to an Atari ROM name from the Arcade Learning Environment.
To get a list of all available games, run the following in a Python script:

```python
import atari_py
print(sorted(atari_py.list_games()))
```

If you are interested in using other
[Gym](https://gym.openai.com/)
environments or your own custom environments, you can make minor modifications to the `main` function in `run_dqn.py`.
Make sure to also adjust any data preprocessing, etc., elsewhere in the code as needed.


### Original DQN

We also include a reference DQN implementation that follows identical experiment procedures to the DeepMind Nature paper.
This will be much slower than our concurrent/synchronized version but can be used if high-fidelity DQN results are needed (e.g. as a baseline in a research paper).

`python run_dqn.py --game=pong`

Note that this command is **not** equivalent to
`run_fast_dqn.py` with 1 worker and concurrency/synchronization disabled (due to how the worker temporarily buffers experiences).


## How It Works

DQN originally follows an *alternating* training mode:
1. Execute 4 actions in the environment.
1. Conduct 1 training minibatch update.
1. Repeat.

For heterogeneuous (CPU + GPU) systems, this strategy is inefficient;
either the CPU or GPU is idle at any given time.
The implementation here introduces two major changes to resolve this.

- **Concurrent Training.**
Rather than alternating between execution and training, these two tasks are performed *concurrently*.
This is possible by selecting greedy actions using the target network while training the main network in a separate thread.
When the CPU is updating the environment, the GPU can be processing the training minibatches.

- **Synchronized Multi-Threaded Execution.**
As done by many deep RL methods like
[A3C](https://arxiv.org/abs/1602.01783),
multiple threads can be used to sample from parallel environment instances and increase overall throughput.
Our implementation executes threads *synchronously* and batches their Q-value predictions together to utilize the GPU more efficiently.
This also reduces I/O competition when sharing a single GPU between many CPU threads.


## License

The code in this repository is available for both free and commerical use under the [MIT License](./LICENSE).
