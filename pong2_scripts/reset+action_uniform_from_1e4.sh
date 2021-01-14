CUDA_VISIBLE_DEVICES=$1 python main.py \
	--id=$0.$2 \
	--env-type=atari \
	--game=pong \
	--count-base-bonus=0.01 \
        --seed=$2 \
	--deploy-policy=reset_policy \
	--switch-memory-priority=0 \
	--switch-sample-strategy=uniform \
	--switch-memory-capcacity=10000 \
	--switch-bsz=512 \
	--policy-diff-threshold=0.5 \
        --result-dir=/pvc/rainbow_lyf/results \
	--checkpoint-interval=100000
