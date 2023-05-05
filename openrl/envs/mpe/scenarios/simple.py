from typing import Optional
import numpy as np
from openrl.envs.mpe.core import Agent, Landmark, World
from openrl.envs.mpe.scenario import BaseScenario

class Scenario(BaseScenario):
    def __init__(self):
        self.render_mode = None

    def make_world(
        self,
        render_mode: Optional[str] = None,
        world_length: int = 25,
        num_agents: int = 3,
        num_landmarks: int = 3,
    ):
        self.render_mode = render_mode
        world = World()
        world.name = "simple"
        world.world_length = world_length
        world.dim_c = 2
        world.num_agents = num_agents
        world.num_landmarks = num_landmarks
        # add agents
        world.agents = [Agent() for i in range(world.num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False
            agent.silent = True
        # add landmarks
        world.landmarks = [Landmark() for i in range(world.num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world, np_random: Optional[np.random.Generator] = None):
        # random properties for agents
        if np_random is None:
            np_random = np.random.default_rng()
        world.assign_agent_colors()
        world.assign_landmark_colors()
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np_random.uniform(-1,+1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = 0.8*np_random.uniform(-1,+1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def reward(self, agent, world):
        dist2 = np.sum(np.square(agent.state.p_pos - world.landmarks[0].state.p_pos))
        return -dist2

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + entity_pos)
