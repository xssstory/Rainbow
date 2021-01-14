CUDA_VISIBLE_DEVICES=$1 python main.py \
	--id=$0.$2 \
	--env-type=atari \
	--game=pong \
	--count-base-bonus=0.01 \
        --seed=$2 \
	--deploy-policy=reset_feature \
	--switch-memory-priority=1 \
	--switch-sample-strategy=recent \
	--switch-memory-capcacity=1000000 \
	--switch-bsz=512 \
	--feature-threshold=0.98 \
        --result-dir=/pvc/rainbow_lyf/results \
	--checkpoint-interval=100000
