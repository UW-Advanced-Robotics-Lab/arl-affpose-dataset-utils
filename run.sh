#!/bin/sh

SRC_CONTAINER=/ARLAffPoseDatasetUtils
SRC_HOST="$(pwd)"

docker run \
    --name arlaffposedatasetutils \
    -it --rm \
    --volume=$SRC_HOST:$SRC_CONTAINER:rw \
    arlaffposedatasetutils
