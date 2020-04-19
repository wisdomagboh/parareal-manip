# Parareal for Robotic Manipulation

A key component of many robotics model-based planning and control algorithms is physics predictions, that is, forecasting a sequence of states given an initial state and a sequence of controls. This process is slow and a major computational bottleneck for robotics planning algorithms.

Parallel-in-time integration methods such as Parareal can help to leverage parallel computing to accelerate physics predictions and thus planning.

We propose combining a coarse (i.e. computationally cheap but not very accurate) predictive physics model, with a fine (i.e. computationally expensive but accurate) predictive physics model, to generate a hybrid model that is at the required speed and accuracy for a given manipulation task.

The Parareal algorithm iterates between a coarse serial integrator and a fine parallel integrator. A key
challenge is to devise a coarse level model that is computationally cheap but accurate enough for Parareal
to converge quickly. 

We propose two coarse physics models for robotic pushing --- An analytical and a deep neural network physics model. We use the Mujoco physics engine as the fine model. 

Here, we provide the source code for our implementation. 

More information can be found in our papers [ISRR 2019](https://arxiv.org/abs/1903.08470) and [JCVS 2020](https://arxiv.org/abs/1912.05958) (Conditionally Accepted)

<img src="analytical_coarse.png" width="204" height="95"/>   <img src="box_center_analytical.png" width="188" height="113"/> <img src="learned_coarse.png" width="175" height="142"/>   <img src="learned_two_sliders.png" width="143" height="105"/> 

## Getting Started

### Create a virtual environment (Code was tested with Ubuntu 16.04 and python3.5)
$ virtualenv -p /usr/bin/python3.5 venv

### Install Physics Simulator Mujoco and dm_control in virtual env 
Please follow instructions from Deepmind's dm_control project [here](https://github.com/deepmind/dm_control).

### Install other required python packages 
$ pip install numpy pandas pyquaternion shapely matplotlib IPython tensorflow==2.0.0-beta1 pillow

### Run setup.py to place custom domains into 'suite'
$ python3.5 setup/setup.py 

### Generate new neural network weights / Use existing 

#### Generate data to train neural network 
$ python3.5 data_generator.py 

Outputs:
'discrete_' files - used for training a model, and created by handing input states and actions from a discrete space to the generator.
'random_' files   - used for testing a model, and created by handing random (uniform) input states and actions in a given range to the generator.
'_initial' files  - contain initial state data for the sliders and the applied action (assume pusher starts at (0,0), and slider 1 starts on x-axis).
'_final' files    - contain displacement data (x,y,theta) and final velocities of the sliders.

#### Train neural network 
$ python3.5 train_model.py

Uses files from previous step and outputs:
model_weights.h5  - the weights of a trained model that can predict for up to 4 sliders. 
					Network configuration: 25(=input),512,256,128,64,24(=output), activation=relu, use_bias=True

#### Test neural network model 
$ python3.5 test_model.py 	  

Uses trained model and test data generated in previous steps and computes prediction error. Also renders predicted final state vs. correct state.

## Running experiments, training and testing

 
## Citation
If you find the code useful please consider citing [ISRR 2019](https://arxiv.org/abs/1903.08470) 

@inproceedings{agboh_isrr19,
  author    = {Wisdom C. Agboh and
               Daniel Ruprecht and
               Mehmet R. Dogar},
  title     = {Combining Coarse and Fine Physics for Manipulation using Parallel-in-Time
               Integration},
  journal   = {International Symposium on Robotics Research},
  year      = {2019}
}

**and** [JCVS 2020](https://arxiv.org/abs/1912.05958) (Conditionally accepted)

@article{agboh_jcvs20,
  author    = {Wisdom C. Agboh and
               Oliver Grainger and 
               Daniel Ruprecht and
               Mehmet R. Dogar},
  title     = {Parareal with a Learned Coarse Model for Robotic Manipulation},
  journal   = {Journal of Computing and Visualization in Science},
  year      = {2020}
}

## Watch a video

[<img src="pusher_slider.png" width="280" height="200"/>](https://youtu.be/5e9oTeu4JOU) [<img src="planning_with_learned_model.jpg" width="280" height="200"/>](https://youtu.be/wCh2o1rf-gA)

## Have a question?
For all queries please contact Wisdom Agboh (wisdomagboh@gmail.com).

## License
This project is licensed under the MIT License - see the 
[LICENSE.md](LICENSE.md) file for details.
