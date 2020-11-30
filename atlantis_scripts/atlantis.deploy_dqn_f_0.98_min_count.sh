python main.py \
	--id=atlantis.deploy_dqn_f0.98_min_count \
	--env-type=atari \
	--game=atlantis \
	--deploy-policy=dqn-feature-min \
	--feature-threshold=0.98 \
    --min-interval=10000 \
	--count-base-bonus=0.01 \
	--checkpoint-interval=100000
