# Legged  Robots  that  Keep  on  Learning


Official codebase for [Legged Robots that Keep on Learning: Fine-Tuning Locomotion Policies in the Real World](https://arxiv.org/abs/2110.05457), which contains code for training a simulated or real A1 quadrupedal robot to imitate various reference motions, pre-trained policies, and example training code for learning the policies.

<p align="center">
   <a href="https://youtu.be/1EUQD7nYfLM">
        <img src="https://github.com/lauramsmith/fine-tuning-locomotion/blob/main/motion_imitation/data/park_pacing.gif" alt="animated" />
   </a>
</p>

Project page: https://sites.google.com/berkeley.edu/fine-tuning-locomotion

## Getting Started

-   Install MPC extension (Optional) `python3 setup.py install --user`

Install dependencies:

-   Install MPI: `sudo apt install libopenmpi-dev`
-   Install requirements: `pip3 install -r requirements.txt`

## Training Policies in Simulation

To train a policy, run the following command:

```
python3 motion_imitation/run_sac.py \
--mode train \
--motion_file [path to reference motion, e.g., motion_imitation/data/motions/pace.txt] \
--int_save_freq 1000 \
--visualize
```

-   `--mode` can be either `train` or `test`.
-   `--motion_file` specifies the reference motion that the robot is to imitate (not needed for training a reset policy).
    `motion_imitation/data/motions/` contains different reference motion clips.
-   `--int_save_freq` specifies the frequency for saving intermediate policies
    every n policy steps.
-   `--visualize` enables visualization, and rendering can be disabled by
    removing the flag.
-   `--train_reset` trains a reset policy, otherwise imitation policies will be trained according to the reference motions passed in.
-   adding `--use_redq` uses REDQ, otherwise vanilla SAC will be used.
-   the trained model, videos, and logs will be written to `output/`.

## Evaluating and/or Fine-Tuning Trained Policies

We provide checkpoints for the pre-trained models used in our experiments in `motion_imitation/data/policies/`.

### Evaluating a Policy in Simulation

To evaluate individual policies, run the following command:
```
python3 motion_imitation/run_sac.py \
--mode test \
--motion_file [path to reference motion, e.g., motion_imitation/data/motions/pace.txt] \
--model_file [path to imitation model checkpoint, e.g., motion_imitation/data/policies/pace.ckpt] \
--num_test_episodes [# episodes to test] \
--use_redq \
--visualize
```

-   `--motion_file` specifies the reference motion that the robot is to imitate
    `motion_imitation/data/motions/` contains different reference motion clips.
-   `--model_file` specifies specifies the `.ckpt` file that contains the trained model
    `motion_imitation/data/policies/` contains different pre-trained models.
-   `--num_test_episodes` specifies the number of episodes to run evaluation for
-   `--visualize` enables visualization, and rendering can be disabled by removing the flag.

### Autonomous Training using a Pre-Trained Reset Controller

To fine-tune policies autonomously, add a path to a trained reset policy (e.g., `motion_imitation/data/policies/reset.ckpt`) and a (pre-trained) imitation policy.

```
python3 motion_imitation/run_sac.py \
--mode train \
--motion_file [path to reference motion] \
--model_file [path to imitation model checkpoint] \
--getup_model_file [path to reset model checkpoint] \
--use_redq \
--int_save_freq 100 \
--num_test_episodes 20 \
--finetune \
--real_robot
```
-   adding `--finetune` performs fine-tuning, otherwise hyperparameters for pre-training will be used.
-   adding `--real_robot` will run training on the real A1 (see [below](#running-mpc-on-the-real-a1-robot) to install necessary packages for running the real A1).     If this is omitted, training will run in simulation.

To run two SAC trainers, one learning to walk forward and one backward, add a reference and checkpoint for another policy and use the `multitask` flag.

```
python motion_imitation/run_sac.py \
--mode train \
--motion_file motion_imitation/data/motions/pace.txt \
--backward_motion_file motion_imitation/data/motions/pace_backward.txt \
--model_file [path to forward imitation model checkpoint] \
--backward_model_file [path to backward imitation model checkpoint] \
--getup_model_file [path to reset model checkpoint] \
--use_redq \
--int_save_freq 100 \
--num_test_episodes 20 \
--real_robot \
--finetune \
--multitask
```

## Running MPC on the real A1 robot

Since the [SDK](https://github.com/unitreerobotics/unitree_legged_sdk) from
Unitree is implemented in C++, we find the optimal way of robot interfacing to
be via C++-python interface using pybind11.

### Step 1: Build and Test the robot interface

To start, build the python interface by running the following: `bash cd
third_party/unitree_legged_sdk mkdir build cd build cmake .. make` Then copy the
built `robot_interface.XXX.so` file to the main directory (where you can see
this README.md file).

### Step 2: Setup correct permissions for non-sudo user

Since the Unitree SDK requires memory locking and high-priority process, which
is not usually granted without sudo, add the following lines to
`/etc/security/limits.conf`:

```
<username> soft memlock unlimited
<username> hard memlock unlimited
<username> soft nice eip
<username> hard nice eip
```

You may need to reboot the computer for the above changes to get into effect.

### Step 3: Test robot interface.

Test the python interfacing by running: 'sudo python3 -m
motion_imitation.examples.test_robot_interface'

If the previous steps were completed correctly, the script should finish without
throwing any errors.

Note that this code does *not* do anything on the actual robot.

## Running the Whole-body MPC controller

To see the whole-body MPC controller in sim, run: `bash python3 -m
motion_imitation.examples.whole_body_controller_example`

To see the whole-body MPC controller on the real robot, run: `bash sudo python3
-m motion_imitation.examples.whole_body_controller_robot_example`
