CUDA_VISIBLE_DEVICES=$1 python main.py \
	--id=road_runner.deploy_whenreset_count_seed_$2 \
	--env-type=atari \
	--game=road_runner \
	--count-base-bonus=0.01 \
        --seed=$2 \
	--deploy-policy=reset \
	--result-dir=/pvc/rainbow_lyf/results \
	--checkpoint-interval=100000
