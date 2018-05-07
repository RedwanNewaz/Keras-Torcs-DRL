# Keras-Torcs-DRL

This repository implements gymTorcs in such a way that support [keras-rl](https://github.com/keras-rl/keras-rl).
Unlike using a specific DRL algorithm such that [DDPG](https://github.com/yanpanlau/DDPG-Keras-Torcs), one can play with many deep RL algorithms here. Tensorboard and ModelCheckpoints are also included in the basic implementation. It is now easier to see the training log online and save the weights periodically. Hyperparameters can easily be setup through the param.yaml file. This feature is very handy while using pycahrm [remote interpreter](https://www.jetbrains.com/help/pycharm/configuring-remote-interpreters-via-ssh.html), no need to go through the whole script.
