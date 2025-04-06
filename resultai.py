import gymnasium as gym
import os
from stable_baselines3 import PPO

# Load the trained model
MODEL_PATH = "models/taxi_ppo_model.zip"

def test_trained_model(model_path, episodes=5, render=False):
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        return
    
    env = gym.make("Taxi-v3")
    model = PPO.load(model_path, env=env)
    
    for episode in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)  # Convert the action to an integer
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")
    
    env.close()

if __name__ == "__main__":
    test_trained_model(MODEL_PATH)
