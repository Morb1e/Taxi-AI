import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("Taxi-v3")
model = PPO("MlpPolicy", env, verbose=1)
model.learn(10000)
model.save("models/taxi_ppo_model")

total_reward = 0
for i in range(10):
    obs, _ = env.reset()
    episode_reward, done = 0, False
    while not done:
        action = int(model.predict(obs, True)[0])
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode_reward += reward
    total_reward += episode_reward
    print(f"Эпизод {i+1}: {episode_reward}")

print(f"Средняя награда: {total_reward / 10}")
env.close()