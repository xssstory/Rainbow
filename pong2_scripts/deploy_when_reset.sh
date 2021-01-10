CUDA_VISIBLE_DEVICES=$1 python main.py \
	--id=pong.deploy_whenreset_count_seed_$2 \
	--env-type=atari \
	--game=pong \
	--count-base-bonus=0.01 \
        --seed=$2 \
	--deploy-policy=reset \
	--checkpoint-interval=100000
