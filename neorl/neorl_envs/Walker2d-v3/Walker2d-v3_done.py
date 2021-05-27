import numpy as np

def get_done(data):

    obs = data["obs"]
    action = data["action"]
    obs_next = data["next_obs"]

    singel_done = False
    if len(obs.shape) == 1:
        singel_done = True
        obs = obs.reshape(1, -1)
    if len(action.shape) == 1:
        action = action.reshape(1, -1)
    if len(obs_next.shape) == 1:
        obs_next = obs_next.reshape(1, -1)

    if isinstance(obs, np.ndarray):
        array_type = np
    else:
        import torch
        array_type = torch

    min_z, max_z = (0.8, 2.0)
    min_angle, max_angle = (-1.0, 1.0)
    min_state, max_state = (-100.0, 100.0)
    
    z = obs_next[:, 1:2]
    angle = obs_next[:, 2:3]
    state = obs_next[:, 3:]
    
    healthy_state = array_type.all(array_type.logical_and(min_state < state, state < max_state), axis=-1, keepdim=True)
    healthy_z = array_type.logical_and(min_z < z, z < max_z)
    healthy_angle = array_type.logical_and(min_angle < angle, angle < max_angle)
    is_healthy = array_type.logical_and(array_type.logical_and(healthy_state, healthy_z), healthy_angle)
    done = array_type.logical_not(is_healthy)

    if singel_done:
        done = done
    else:
        done = done.reshape(-1, 1)
        
    return done