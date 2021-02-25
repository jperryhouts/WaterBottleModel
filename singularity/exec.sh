#!/bin/bash -e

cmd="$1"
shift

if [ "$cmd" == "compile" ]; then
    cmake "${@}"
    make -j4
elif [ "$cmd" == "exec" ]; then
    ${@}
else
    echo "Command not recognized: '${cmd}'"
    exit 1
fi
