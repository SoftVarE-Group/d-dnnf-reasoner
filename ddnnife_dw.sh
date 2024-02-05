#!/bin/bash
# A script to wrap the docker container.
# This simplifies the usage, due to hidding the building of the image and execution of the container.
# Further, verbose parameters such as adding a volume are hidden, resulting in easier accesibility for the user.

if [ $# -eq 0 ]
then
    echo "USAGE: ./ddnnife_dw.sh <rebuild | [VOLUME FOLDER] [ddnnife PARAMETERS; VOLUME FOLDER IS MOUNTED TO \"internal/\"]>"
    echo -e "\trebuild: Rebuilds the docker image if there already exists one."
    echo -e "\t[VOLUME FOLDER]: The folder and its files that should be available to the container."
    echo -e "\tExample: ./ddnnife_dw.sh example_input internal/auto1.cnf count-features internal/res"
    exit 0
fi

if [[ "$(docker images -q ddnnife 2> /dev/null)" == "" ]] || [ "$1" == "rebuild" ]
then
    echo -e "Building the docker image. Make sure that the current folder contains ddnnife as well as its Dockerfile...\n"
    docker build --platform linux/amd64 -t ddnnife .
fi

if [ "$1" == "rebuild" ]
then
    exit 0
fi

docker run --platform linux/amd64 -it --rm -v $(cd "$(dirname "$1")"; pwd)/$(basename "$1"):/internal ddnnife ${@:2}