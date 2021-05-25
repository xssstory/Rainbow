CUDA_VISIBLE_DEVICES=$1 python main.py \
	--id=$0.$2 \
	--env-type=atari \
	--game=battle_zone \
	--count-base-bonus=0.01 \
        --seed=$2 \
	--deploy-policy=fixed \
	--delploy-interval=700 \
	--result-dir=/pvc/rainbow_lyf/results \
	--checkpoint-interval=100000
