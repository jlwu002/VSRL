# Verified Safe Reinforcement Learning for Neural Network Dynamic Models (NeurIPS 2024)

1. Run `generate_grid.py` to generate the grid for verification.  
2. Train a vanilla controller by setting **Line 170** in `moving_obs/train.py` to `False` (`use_reachability = False`), and comment out **Line 149** (`ppo_agent.load`) in `train.py`.  
3. Train with bounds by setting **Line 170** to `True` and loading the checkpoint from the vanilla controller (`ppo_agent.load`) .  
4. The controller for each **k-th step reachability safety** will be stored in the `outputs` folder. If a controller is not fully verified for a given **k**, the filename will have the suffix `_not_verified.pth`.  
5. If you observe a significant decrease in reward or reach the target verification step, stop training. Run `check_collide.py` to verify the safety of the desired input region.  
6. Based on the results from `check_collide.py` (modify `target_steps`), split the input region and continue from **Step 3** for each input region cluster (load from a selected checkpoint).  
7. Stop if all input regions are verified safe for the corresponding controller.  
