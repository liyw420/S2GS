
DATA_ROOT_DIR="/media/vincent/HDD-01/S2GS/data"
DATA_OUTPUT_DIR="/media/vincent/HDD-01/S2GS/output"

DATASETS=(
    technicolor
    )

SCENES=(
    birthday
    # painter
    # train
    # theater
    # remy
    )

RESOLUTION=(
            # 1
            2
            # 4
            # 8
            )

for SCENE in "${SCENES[@]}"; do

# ----- (1) Processing Data Following the Steps of 3DGStream -----
    SCENE_PATH=${DATA_ROOT_DIR}/${DATASETS}/${SCENE}
    OUTPUT_PATH=${DATA_OUTPUT_DIR}/${DATASETS}/${SCENE}

    CMD_1="python ./scripts/pre_technicolor/video.py \
    --source ${SCENE_PATH} \
    "

    # ----- (2) Processing Data from 3DGStream to Queen -----
    CMD_2="python ./scripts/pre_technicolor/downsample_point.py \
    ${SCENE_PATH}/mvs_input_0.0.ply \
    ${SCENE_PATH}/points3D_downsample2.ply \
    "

    # # ----- (3) Delete Files to Save Storage  -----
    # CMD_9="rm -f ${SCENE_PATH}/*.mp4 && rm -rf ${SCENE_PATH}/frame000{001..300} ${SCENE_PATH}/images ${SCENE_PATH}/distorted"

    # ----- (4) Train, Render and Metrics  -----
    CMD_3="python ./train.py \
    --config configs/${DATASETS}_${SCENE}.yaml \
    -s ${SCENE_PATH} \
    -m ${OUTPUT_PATH} \
    --log_images \
    "

    CMD_4="python ./metrics_video.py \
    -m ${OUTPUT_PATH} \
    "

    echo "========= ${SCENE}: XXX   ========="
    # eval $CMD_1
    # eval $CMD_2
    eval $CMD_3
    # eval $CMD_4

done