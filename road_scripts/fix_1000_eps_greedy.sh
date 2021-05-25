CUDA_VISIBLE_DEVICES=$1 python main.py \
	--id=$0.$2 \
	--env-type=atari \
	--game=road_runner \
	--count-base-bonus=0.01 \
	--explore-eps=(250e3,0.01) \
        --seed=$2 \
	--deploy-policy=fixed \
	--delploy-interval=1000 \
	--result-dir=/pvc/rainbow_lyf/results \
	--checkpoint-interval=100000
