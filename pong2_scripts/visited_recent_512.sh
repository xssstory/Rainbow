CUDA_VISIBLE_DEVICES=$1 python main.py \
	--id=$0.$2 \
	--env-type=atari \
	--game=pong \
	--count-base-bonus=0.01 \
        --seed=$2 \
	--deploy-policy=visited \
	--switch-memory-priority=0 \
	--switch-sample-strategy=recent \
	--switch-memory-capcacity=1000000 \
	--switch-bsz=512 \
        --result-dir=/pvc/rainbow_lyf/results \
	--checkpoint-interval=100000