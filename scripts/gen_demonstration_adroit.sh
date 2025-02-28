# bash scripts/gen_demonstration_adroit.sh door
# bash scripts/gen_demonstration_adroit.sh hammer
# bash scripts/gen_demonstration_adroit.sh pen

#cd third_party/VRL3/src

task=${1}

CUDA_VISIBLE_DEVICES=0 python third_party/VRL3/src/gen_demonstration_expert.py --env_name $task \
                        --num_episodes 10 \
                        --root_dir "/data/code/FlowPolicy/data/" \
                        --expert_ckpt_path "/data/code/FlowPolicy/third_party/VRL3/vrl3_ckpts/vrl3_${task}.pt" \
                        --img_size 84 \
                        --not_use_multi_view \
                        --use_point_crop
