CUDA_VISIBLE_DEVICES=$1 python main.py \
	--id=beam_rider.deploy_whenreset_count_seed_$2 \
	--env-type=atari \
	--game=beam_rider \
	--count-base-bonus=0.01 \
        --seed=$2 \
	--deploy-policy=reset \
	--result-dir=/pvc/rainbow_lyf/results \
	--checkpoint-interval=100000