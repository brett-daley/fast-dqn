
def DeepQNetwork(*args, pytorch=False, **kwargs):
    if pytorch:
        from fast_dqn.deep_q_network.pytorch import PytorchDeepQNetwork
        dqn_cls = PytorchDeepQNetwork
    else:
        from fast_dqn.deep_q_network.tensorflow import TensorflowDeepQNetwork
        dqn_cls = TensorflowDeepQNetwork

    return dqn_cls(*args, **kwargs)
