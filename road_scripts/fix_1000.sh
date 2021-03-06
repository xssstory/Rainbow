CUDA_VISIBLE_DEVICES=$1 python main.py \
	--id=road_runner.deploy_1000_count_seed_$2 \
	--env-type=atari \
	--game=road_runner \
	--count-base-bonus=0.01 \
        --seed=$2 \
	--deploy-policy=fixed \
	--delploy-interval=1000 \
	--checkpoint-interval=100000
