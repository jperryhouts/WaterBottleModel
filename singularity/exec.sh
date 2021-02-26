#!/bin/bash -e

cmd="$1"
shift

print_help() {
    echo "
Usage: utils.sif [-h|--help] <cmd> [<args>]

Options
    -h, --help
        Print this message and exit

Commands
    compile
        Compile the model into the current directory. You would usually call
        this command from a different directory, to avoid cluttering your
        workspace with build files.

        It can be followed by any arguments that you'd like to pass to cmake,
        but the last argument should be the path to the source directory.

    exec
        Execute the remaining arguments as a shell command inside the container.

Examples
    Set up your PATH to find the utils.sif executable:

        curdir=\"\$(pwd)\"
        echo \"export PATH=\${curdir}:\\\$PATH\" >> ~/.bashrc
        source ~/.bashrc

    Compile the code in a new directory, then run an example model:

        mkdir build ; cd build
        utils.sif compile ../src/
        cd ../examples
        utils.sif exec ../build/wbm waterbottlemodel.prm

    Compile in optimized mode (by default, the code will compile in 'debug'
    mode, which will run more slowly, but will also print more helpful error
    messages):

        utils.sif compile -DCMAKE_BUILD_TYPE=Release ../src/

    Run the model using multiple cores (Note, I've only been able to get this
    working using multiple cores on a single node). Replace the '-np 8' with
    the number of cores on your machine:

        utils.sif exec mpirun -np 8 ../build/wbm model.prm
"
}

case "$cmd" in
    '-h'|'--help')
        print_help
        exit 0
        ;;
    "compile")
        cmake "${@}"
        make -j4
        ;;
    "exec")
        "${@}"
        ;;
    *)
        echo "Command not recognized: ${cmd}"
        echo "Try --help for explanation of available commands."
        exit 1
        ;;
esac
