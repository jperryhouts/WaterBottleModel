## Water Bottle Model for lower crustal flow

This is standalone modeling software for simulating flow in a viscous lower crust, below an elastic upper plate.
The structure is intentionally modular to allow this numerical model to be incorporated into existing modeling frameworks.

### Singularity image

This software requires the Deal.II library be installed on your machine, but a simpler way to get going quickly is with the Singularity containerization software.

You can download a [pre-built image](https://sse5tbvwkjja2.s3.amazonaws.com/utils.sif) based on the definition file in the singularity directory in this repository. The container bundles an environment containing everything you need to compile and run this code.

In order to use it, you will need to have Singularity installed on your computer or compute cluster. You can set up Singularity by following the instructions [here](https://sylabs.io/guides/3.6/user-guide/quick_start.html#quick-installation-steps). The following example demonstrates how to use the container to compile and run the water bottle model as follows:

```
git clone https://github.com/jperryhouts/WaterBottleModel.git
cd WaterBottleModel
wget https://sse5tbvwkjja2.s3.amazonaws.com/utils.sif
mkdir build && cd build
../utils.sif compile ../src/
cd ../examples
../utils.sif exec mpirun -np 2 ../build/wbm waterbottlemodel.prm
```
