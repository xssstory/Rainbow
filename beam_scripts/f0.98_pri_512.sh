CUDA_VISIBLE_DEVICES=$1 python main.py \
	--id=$0.$2.out \
	--env-type=atari \
	--game=beam_rider \
	--deploy-policy=dqn-feature \
	--feature-threshold=0.98 \
    --switch-memory-priority=True \
	--switch-bsz=512 \
    --seed=$2 \
	--count-base-bonus=0.01 \
	--checkpoint-interval=100000
