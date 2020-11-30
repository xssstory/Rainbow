CUDA_VISIBLE_DEVICES=1 python main.py \
	--id=enduro.deploy_dqn_f0.97_count \
	--env-type=atari \
	--game=enduro \
	--deploy-policy=dqn-feature \
	--feature-threshold=0.97 \
	--count-base-bonus=0.01 \
	--checkpoint-interval=100000
