CUDA_VISIBLE_DEVICES=$1 python main.py \
	--id=$0.$2 \
	--env-type=atari \
	--game=qbert \
	--count-base-bonus=0.01 \
        --seed=$2 \
	--deploy-policy=dqn-feature \
	--switch-memory-priority=0 \
	--switch-sample-strategy=uniform \
	--switch-memory-capcacity=10000 \
	--switch-bsz=512 \
	--feature-threshold=0.98 \
        --result-dir=/pvc/rainbow_lyf/results \
	--use-gradient-weight \
	--adaptive-softmax \
	--checkpoint-interval=100000
