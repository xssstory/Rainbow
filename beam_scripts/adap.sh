CUDA_VISIBLE_DEVICES=$1 python main.py \
	--id=beam_rider.deploy_adaptive.100_count_seed_$2 \
	--env-type=atari \
	--game=beam_rider \
	--deploy-policy=dqn-feature-min \
	--min-interval=adaptive.100 \
	--feature-threshold=2 \
	--count-base-bonus=0.01 \
        --seed=$2 \
	--checkpoint-interval=100000
