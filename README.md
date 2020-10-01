# Data driven discretizations for solving 2D PDEs  (Luciano's bootleg)

This repository explores extensions of the techniques developed in:

>  [Learning data-driven discretizations for partial differential equations](https://www.pnas.org/content/116/31/15344).
  Yohai Bar-Sinai\*, Stephan Hoyer\*, Jason Hickey, Michael P. Brenner.
  PNAS 2019, 116 (31) 15344-15349.

See [this repository](https://github.com/google/data-driven-discretization-1d)
for the code used to produce results for the PNAS paper.

This is not an official Google product.

# Installation

Installation is most easily done using pip.
1. Create or activate a virtual environment (e.g. using `virtualenv` or `conda`).
2. [Install TensorFlow 1.x](https://www.tensorflow.org/install/pip) - use the GPU package if you plan to run on a CUDA device!
3. If you just want to install the package without the code,
   simply use pip to install directly from github:

   `pip install git+https://github.com/lucianodrozda/data-driven-pdes-google`

   If you want to fiddle around with the code, `cd` to where you want to store the code,
  clone the repo and install:
```bash
cd <your directory>
git clone git+https://github.com/lucianodrozda/data-driven-pdes-google
pip install -e data-driven-pdes
```

# Usage

We aim to make the code accessible for researchers who want to apply our method to their favorite PDEs. To this end we wrote, and continue to write, tutorials and documentation.
This is still very much in development, please [open an issue](https://github.com/google-research/data-driven-pdes/issues) if you have questions.

1. [A tutorial notebook](tutorial/Tutorial.ipynb) that explains some of the basic notions in the code base and demonstrates how to use the framework to define new equations.
2. [This notebook](tutorial/advection_1d.ipynb) contains a complete example of creating a training database, defining a model, training it and evaluating the trained model (well documented, though less pedagogical).

# Luciano's contributions
The code now provides spatial discretizations for the 2D compressible Euler equations following the Jameson-Schmidt-Turkel (JST) and Component- and Characteristic-wise 5th order WENO schemes. Random flow states from isentropic vortex convection, circular shock, circular shock and vortex interaction, and 1D acoustic gaussian wave to double shear layer can be obtained.

When instantiating neural net models with Euler equations, please consider using the `FiniteVolumeEuler` class within `datadrivenpdes/euler/equations.py`, since the `FiniteDifferenceEuler` one does not support Strong Stability Preserving 3rd order Runge-Kutta time-stepping scheme yet.

Scripts showing how to create training data, run training, and evaluate performance (accuracy-runtime tradeoff) can be found in `datadrivenpdes/my-pipelines/euler`.

Runge-Kutta 3-stepped scheme implementation in the `FiniteVolumeEuler` class (proper to neural net models) led to problems when using the `.fit()` method during training. The neural net model was trying to perform operations between a `tf.EagerTensor` and a `tf.Tensor`, which is forbidden. 
The solution was to recode the training loop in Eager mode following instructions [here](https://www.tensorflow.org/guide/eager#eager_training).
An end-to-end example of low-level handcrafted training loop can be found [here](https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch#using_the_gradienttape_a_first_end-to-end_example).

An implementation of the Turing equations is also available.

Finally, only periodic boundary conditions are supported.
