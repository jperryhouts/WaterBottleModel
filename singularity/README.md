## Singularity container

Singularity is a containerization tool that makes it possible to compile and run code in a controlled compute environment, independent of the configuration of your machine.

This container does not contain the compiled model, only a build/run environment that is pre-configured for compiling and executing the model code. This allows further development of the code base without the need to install the Deal.II library on your machine. It also does not require root user privileges, so long as Singularity is already installed (as is the case on many compute clusters).

You can download a pre-built singularity file, which bundles the entire container [here](https://sse5tbvwkjja2.s3.amazonaws.com/utils.sif).

If you have root user privileges on your machine, you can set up Singularity by following the instructions [here](https://sylabs.io/guides/3.6/user-guide/quick_start.html#quick-installation-steps). You can then re-build this container using the command:

    sudo singularity build utils.sif utils.def

However, since the container only includes a build environment, it's unlikely that you'll need to re-build it yourself.
