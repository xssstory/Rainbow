python main.py \
	--id=atlantis.deploy_adaptive.1000_count \
	--env-type=atari \
	--game=atlantis \
	--deploy-policy=dqn-feature-min \
	--min-interval=adaptive.500 \
	--feature-threshold=2 \
	--count-base-bonus=0.01 \
	--checkpoint-interval=100000
