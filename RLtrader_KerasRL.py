from fileinput import filename
import os
import pandas as pd
import numpy as np
from enviroments.rl_env import RLEnvSpot
import random
import tensorflow as tf
from tensorflow.keras.optimizers.legacy import Adam
from rl.agents import DQNAgent
from rl.policy import EpsGreedyQPolicy, BoltzmannQPolicy
from rl.memory import SequentialMemory
from sklearn import preprocessing

from enviroments.rl_env import RLEnv
import TA_tools

def build_model(states, actions, layers_set, window_length):
        model = tf.keras.models.Sequential()
        #model.add(tf.keras.layers.Dense(128, activation='relu', input_shape=(window_length,states)))
        model.add(tf.keras.layers.Flatten(input_shape=(window_length,states)))
        #model.add(Reshape((lookback_window_size, states), input_shape=(window_length, lookback_window_size, states)))
        #model.add(LSTM(neurons, return_sequences = True, input_shape=(lookback_window_size, states), activation = 'relu'))
        #model.add(Dense(states, activation='relu'))
        #model.add(Flatten(input_shape=(window_length,lookback_window_size,states)))
        for k in layers_set.values():
            model.add(tf.keras.layers.Dense(k, activation='relu'))
        model.add(tf.keras.layers.Dense(actions, activation='linear'))
        model.add(tf.keras.layers.Flatten())
        #model.summary()
        return model
def build_agent(model, actions, window_length, policy=EpsGreedyQPolicy()):
        #policy = BoltzmannQPolicy()
        test_policy=policy
        memory = SequentialMemory(limit=100*window_length, window_length=window_length)
        dqn = DQNAgent(model=model, memory=memory, policy=policy, test_policy=test_policy,
                       nb_actions=actions, nb_steps_warmup=58_111, target_model_update=1e-3, 
                       enable_double_dqn=True, enable_dueling_network=True, dueling_type='avg')
        return dqn

if __name__=="__main__":
    df = TA_tools.get_df(ticker='BTCTUSD', interval_list=['1m'], type='trader', futures=False, indicator=None, period=None)
    dates_df = df['Opened'].to_numpy()
    df = df.drop(columns='Opened').to_numpy()
    df = TA_tools.add_particular_MAband(df,8,25,65,0.73)
    df = TA_tools.add_particular_MAband(df,0,5,61,0.57) # best RMA parameters
    df = TA_tools.add_particular_MAband(df,0,5,294,0.31) # best RMA parameters
    df = TA_tools.add_particular_MAband(df,1,9,1,1.12) # best SMA 
    df = TA_tools.add_particular_MAband(df,1,4,89,0.48) # best SMA
    df = TA_tools.add_particular_MAband(df,2,6,22,0.39) # best EMA
    df = TA_tools.add_particular_MAband(df,2,5,191,0.41) # best EMA
    df = TA_tools.add_particular_MAband(df,3,8,205,0.47) # best WMA
    df = TA_tools.add_particular_MAband(df,3,6,50,0.45) # best WMA
    df = TA_tools.add_particular_MAband(df,4,5,8,0.57) # best VWMA
    df = TA_tools.add_particular_MAband(df,4,5,212,0.52) # best VWMA
    df = TA_tools.add_particular_MAband(df,5,4,61,0.44) # best SMMA
    df = TA_tools.add_particular_MAband(df,6,4,93,0.47) # best KMA
    df = TA_tools.add_particular_MAband(df,6,4,11,0.59) # best KMA
    df = TA_tools.add_particular_MAband(df,7,6,35,0.82) # best TMA
    df = TA_tools.add_particular_MAband(df,7,9,118,0.72) # best TMA
    df = TA_tools.add_particular_MAband(df,8,24,144,0.64) # best HullMA
    df = TA_tools.add_particular_MAband(df,7,9,118,0.72) # best HullMA
    print(f'len(df[-1,:]) {len(df[-1,:])}')

    policy=BoltzmannQPolicy()

    env = RLEnvSpot(df=df[-58_111:,:], dates_df=dates_df[-58_111:], excluded_left=0, init_balance=600, postition_ratio=1.0, leverage=1, fee=0.0, slippage=0.0001, Render_range=120, visualize=False)
    model = build_model(np.shape(env.reset())[0], 3, {0:128, 1:64, 2:16}, 180)
            #model.summary()
    dqn = build_agent(model, 3, 180, policy=policy)
    dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])
    dqn.fit(env, nb_steps=500_000, visualize=env.visualize, verbose=2, log_interval=1)
    env.visualize = False
    scores = dqn.test(env, nb_episodes=100, visualize=env.visualize)
    
    '''scores_l=[]
    for _ in range(10):
            lookback_size=random.choice(list(range(10,61,1)))
          #lookback_sizes = list(range(1,31,1))
          #for lookback_size in lookback_size:
            env = RLEnvSpot(df=df[-52_945:,:], dates_df=dates_df[-52_945:], excluded_left=0, init_balance=600, postition_ratio=1.0, leverage=1, fee=0.0, slippage=0.0001, Render_range=120, visualize=True)
            #test_env = CustomEnv(df, lookback_window_size=lookback_size, max_steps=50_000, visualize=True, initial_balance=20, init_postition_size=2.0, leverage=125)
            n_count=int((lookback_size*(len(df[0,:]))+3)//4)
            n_counts=list(range(n_count//16,n_count+1,n_count//16))
            n_layers=list(range(3,4))
            layers_set={}
            for i in range(random.choice(n_layers)):
              layers_set[i]=random.choice(n_counts)
            #layers_set={0:3886, 1:3350, 2:536, 3:1742}
            n_windows=random.choice(list(range(1,5)))
            print("####################################################")
            print(f'n_windows: {n_windows}', end='  ')
            #print(f'layers_count: {layers_set.keys()}', end='  ')
            print(f'layers_size: {layers_set.values()}', end='  ')
            print(f'lookback_size: {lookback_size}')
            print(f'np.shape(env.reset())[0] {np.shape(env.reset())[0]} env.action_space.shape {env.action_space.shape}')
            model = build_model(np.shape(env.reset())[0], 3, layers_set, n_windows)
            #model.summary()
            dqn = build_agent(model, 3, n_windows, policy=policy)
            dqn.compile(tf.keras.optimizers.Adam(learning_rate=4e-4), metrics=['mae'])
            ## Callbacks:
            #tfboard = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
            #dqn.fit(train_env, nb_steps=100000, visualize=False, callbacks=[tfboard], verbose=2, log_interval=250)
            dqn.fit(env, nb_steps=30_000, visualize=env.visualize, verbose=2, log_interval=250)
        
            scores = dqn.test(env, nb_episodes=10, visualize=env.visualize)
            print(scores)
            scores_l.append([n_windows,n_layers,n_count,scores])
            cv = lambda x: abs(np.std(x, ddof=1) / np.mean(x)) * 100
            median=np.median(scores.history['episode_reward'])
            mean_r=np.mean(scores.history['episode_reward'])
            zm=cv(scores.history['episode_reward'])
            print(f'n_windows: {n_windows}', end='  ')
            #print(f'layers_count: {layers_set.keys()}', end='  ')
            print(f'layers_size: {layers_set.values()}', end='  ')
            print(f'lookback_size: {lookback_size}')
            print(f'mediana: {median}', end='  ')
            print(f'srednia: {mean_r}', end=' ')
            print(f'wsp. zmiennosci: {zm:.2f}%')
            if zm<=65 and median>1_200 and mean_r>1_200:
              models_dir=f'/content/drive/MyDrive/RLtrader/models/'
              if not os.path.exists(models_dir): os.makedirs(models_dir)
              filepath=f'WspZm-{zm:.2f} Me-{median:.0f} window-{n_windows} lookback-{lookback_size} layers-{list(layers_set.values())} df_cols-{len(df[0])}.h5'
              with open('readme.txt', 'w') as f:
                for col in len(df[0]):
                  f.write(col+' ')
              dqn.save_weights(models_dir+filepath, overwrite=False)
              print(f"#### Zapisano model: {filepath} #####")
            print("####################################################")'''
