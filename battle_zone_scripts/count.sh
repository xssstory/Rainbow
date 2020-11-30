CUDA_VISIBLE_DEVICES=$1 python main.py \
	--id=battle_zone_rider_deploy_none_count_seed_$2 \
	--count-base-bonus=0.01 \
	--env-type=atari \
        --seed=$2 \
	--game=battle_zone \
	--checkpoint-interval=100000 \
