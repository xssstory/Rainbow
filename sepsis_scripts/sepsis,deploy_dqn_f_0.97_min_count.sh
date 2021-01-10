CUDA_VISIBLE_DEVICES=0 python main.py \
--id=sepsis_deploy_f_0.97_min_count \
--env-type=sepsis \
--deploy-policy=dqn-feature-min \
--min-interval=10000 \
--feature-threshold=0.97 \
--count-base-bonus=0.01 \
--T-max=50000000 \
--hidden-size=128 \
--memory-capacity=1000000 \
--target-update=1000 \
--learn-start=1000 \
--evaluation-interval=10000 \
--evaluation-episodes=1000 \
--checkpoint-interval=10000 \
--reward-clip=-1 \
--max-episode-length=50