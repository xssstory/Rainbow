CUDA_VISIBLE_DEVICES=$1 python main.py \
	--id=pong.eval_$2 \
	--env-type=atari \
	--game=pong \
	--count-base-bonus=0.01 \
        --seed=$2 \
	--deploy-policy=reset \
        --evaluate --state-visitation-episodes 10 \
	--model=$3 
