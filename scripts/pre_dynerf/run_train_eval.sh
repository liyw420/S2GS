
DATA_ROOT_DIR="/media/vincent/HDD-01/queen_lightweight/data"
DATA_OUTPUT_DIR="/media/vincent/HDD-01/queen_lightweight/output"

DATASETS=(
    dynerf
    )

SCENES=(
    coffee_martini
    cook_spinach
    cut_roasted_beef
    flame_salmon
    flame_steak
    sear_steak
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

    CMD_1="python ./scripts/pre_dynerf/n3v2blender.py ${SCENE_PATH}"

    CMD_2="python ./scripts/pre_dynerf/video.py \
    --source ${SCENE_PATH} \
    --target ${SCENE_PATH} \
    "

    CMD_3="python ./scripts/pre_dynerf/convert.py \
    -s ${SCENE_PATH}/frame000001 \
    "

    # ----- (2) Processing Data from 3DGStream to Queen -----
    CMD_4="python ./scripts/pre_dynerf/downsample_point.py \
    ${SCENE_PATH}/frame000001/fused.ply \
    ${SCENE_PATH}/points3D_downsample2.ply \
    "

    CMD_5="python ./scripts/pre_dynerf/copy_cams.py \
    --source ${SCENE_PATH}/frame000001 \
    --scene ${SCENE_PATH} \
    "

    CMD_6="python ./scripts/pre_dynerf/convert_frames.py \
    -s ${SCENE_PATH} \
    --resize \
    "

    CMD_7="python ./scripts/pre_dynerf/DataFrom3DGStreamToQueen.py \
    -s ${SCENE_PATH} \
    "

    CMD_8="python ./scripts/pre_dynerf/imgs2poses.py \
    --scenedir ${SCENE_PATH}/frame000001 \
    --outdir ${SCENE_PATH} \
    "

    # ----- (3) Delete Files to Save Storage  -----
    CMD_9="rm -f ${SCENE_PATH}/*.mp4 && rm -rf ${SCENE_PATH}/frame000{001..300} ${SCENE_PATH}/images ${SCENE_PATH}/distorted"

    # ----- (4) Train, Render and Metrics  -----
    CMD_10="python ./train.py \
    --config configs/${DATASETS}_${SCENE}.yaml \
    -s ${SCENE_PATH} \
    -m ${OUTPUT_PATH} \
    --log_images \
    "

    CMD_11="python ./metrics_video.py \
    -m ${OUTPUT_PATH} \
    "

    echo "========= ${SCENE}: XXX   ========="
    # eval $CMD_1
    # eval $CMD_2
    # eval $CMD_3
    # eval $CMD_4
    # eval $CMD_5
    # eval $CMD_6
    # eval $CMD_7
    # eval $CMD_8
    # eval $CMD_9
    eval $CMD_10
    # eval $CMD_11
done