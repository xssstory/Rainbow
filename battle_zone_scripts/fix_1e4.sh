CUDA_VISIBLE_DEVICES=$1 python main.py \
	--id=battle_zone.deploy_10000_count_seed_$2 \
	--env-type=atari \
	--game=battle_zone \
	--count-base-bonus=0.01 \
        --seed=$2 \
	--deploy-policy=fixed \
	--delploy-interval=10000 \
	--checkpoint-interval=100000
