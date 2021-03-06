CUDA_VISIBLE_DEVICES=$1 python main.py \
	--id=beam_rider.deploy_resetpolicy_diffthreshold$3_recent_bsz512_count_seed_$2 \
	--env-type=atari \
	--game=beam_rider \
	--count-base-bonus=0.01 \
        --seed=$2 \
	--deploy-policy=reset_policy \
	--switch-memory-priority=0 \
	--switch-sample-strategy=recent \
	--switch-memory-capcacity=1000000 \
	--switch-bsz=512 \
	--policy-diff-threshold=$3 \
        --result-dir=/pvc/rainbow_lyf/results \
	--checkpoint-interval=100000
