from typing import Dict, Tuple, Any 
from engine.environment.state import GameState 
from engine.agents.agent_data import AgentRole, Teams 
from engine.environment.state_ops import StateOperations as stateops 



class ObservationBuilder:

    VISION_RANGE = 3.0 
    SELF_FEATURE_DIM = 9
    FEATURE_DIM_PER_ENTITY = 5   # NUmber of features per agent 
    TOTAL_PARTIAL_DIM = 20  ## 9 + Total entity slots (3 agents) * 5 features per agent 
    NUM_ROLES = 4

    @staticmethod
    def _extract_self_features(state : GameState, agent_id : int) -> np.ndarrar:

        features = np.zeros(ObservationBuilder.SELF_FEATURE_DIM, dtype = np.float32)

        features[0] = stateops.get_hp_ratio(state, agent_id)  # Total of 9 features with gp, stamina, normalized positions and 3 skills and 2 if blocking or invincible
        features[1] = stateops.get_stamina_ratio(state, agent_id)
        features[2], features[3] = stateops.normalize_dims(state, agent_id)
        features[4] = 1.0 if state.is_blocking[agent_id] else 0.0
        features[5] = 1.0 if state.is_invincible[agent_id] else 0.0 
        features[6] = stateops.is_skill_ready(state, ActionTypes.BASIC)  #Check if basic skill ready
        features[7] = stateops.is_skill_ready(state, ActionTypes.UTILITY)
        features[8] = stateops.is_skill_ready(state, ActionTypes.ULTIMATE)

        return features 

    @staticmethod 
    def _extract_entity_features(state : GameState, self_id : AgentRole, target_id : AgentRole, check_vision : bool) -> Tuple[np.ndarray, AgentRole];

    ## Extract the features of one out of 3 other agents, of size 5 each which contain relative pos and norm hp, stamina and such

    features = np.zeros(ObservationBuilder.FEATURE_DIM_PER_ENTITY, dtype = np.float32)
    role_id = state.roles[target_id]

    if state.hp[target_id] <= 0:
        return features, role_id  # There is no need for obs vuilding for a dead agent teammate or entity

    visible = True ## Vision Verification loop.. true for agents in the same team, but for hero agents and boss is , this is false till we go in range
    if check_vision: ### A way to check if boss in in range. This is true only for agents in hero team and if the target is boss
        visible = False # False till we go in range of boss

        for teammate_id in [AgentRole.TANK, AgentRole.DEALER, AgentRole.HEALER]:
            if state.hp[teammate_id] > 0:  # For alive agents
                dist_between_hero_to_boss = stateops.get_distance(state, teammate_id, target_id)

                if dist <= ObservationBuilder.VISION_RANGE: ## If any hero agent is in range with boss, then the details are visible for all agents
                    visible = True 
                    break 

    if not visible:  # If heroes cannot see the boss, then position and hp are negative to let the network know that the boss is not visible
        features[0:2] -= 1.0 #Obscured position coordinates
        return features, role_id

    features[0] = (state.positions[target_id, 0] - state.positions[self_id, 0])  # calculating relative position
    features[1] = (state.positions[target_id, 1] - state.posiions[self_id, 1]) 
    features[2] = stateops.get_hp_ratio(state, target_id)
    features[3] = stateops.get_stamina_ratio(state, target_id)
    features[4] = 1.0 ## Alive flad confirmation 

    return features, role_id 


@staticmethod 
def build_partial_obs(state : GameState, agent_id : int) -> Tuple[np.ndarray, np.ndarray]:

    continuous_obs = np.zeros(ObservationBuilder.TOTAL_CONTINUOUS_DIM, dtype = np.float32)
    role_ids = np.zeros(NUM_ROLES, dtype = np.int32)   # For Role embeddings maybe.. Contains role IDs of self and other entity IDs..

    role = state.role[agent_id]
    is_boss = (role == AgentRole.BOSS)
    boss_id = AgentRole.BOSS 

    continuous_obs[0:9] = ObservationBuilder._extract_self_features(state, agent_id) ## Populate the arrays with know information first 
    role_ids[0] = role

    external_entities = stateops.get_agents_other_than_self(state, agent_id)  ## Moslty we will get in order of agent order.. so this works

    for slot_idx, target_id in enumerate(external_entities):

        check_vision = (not is_boss and target_id = boss_id)  ## If self is not boss (heroes) and if the other entity is boss then check vision is True

        features, target_role = ObservationBuilder._extract_entity_features(state, agent_id, target_id, check_vision = check_vision)

        start_stride = 9 + (slot_idx * ObservationBuilder.FEATURE_DIM_PER_ENTITY)  # Its so that we can simply start with that entitys space. That is 9 + 5, 9 + 10, 9 + 15
        end_stride = start_stride + ObservationBuilder.FEATURE_DIM_PER_ENTITY  # End stride... to specify where the entitys features end
        continuous_obs[start_stride : end_stride] = features  ## Add already extracted features to the continuous vector at exactly at that position

        role_ids[slot_idx + 1] = target_role   ## Leave the first slot for the self agent and fill other spaces with other entity IDs..

    return continuous_obs, role_ids 


@staticmethod 
def build_full_state(state : GameState) -> Tuple[np.ndarray, np.ndarray]:  ## This is the observation builder for the global state

    num_agents = state.num_agents 
    global_features = np.zeros(num_agents * ObservationBuilder.TOTAL_CONTINUOUS_DIM, dtype = np.float32)  # There is no need for roles IDs here to be seperate, we just need to have a global role ID sequence according to agent index

    for idx in range(num_agents):

        features_start_stride = idx * ObservationBuilder.TOTAL_CONTINUOUS_DIM   # Having 4 times the normal amount, where each agents self comes exactly once. This way we have the whole meaningful information 
        feature_end_stride = feature_start_stride + ObservationBuilder.TOTAL_CONTINUOUS_DIM

        features, roles = ObservationBuilder.build_partial_obs(state, idx)

        global_features[feature_start_stride : feature_end_stride] = features 

    return global_features #, state.roles # So we can send the role array directly. But there is no need as we initialized the same in rollout.py

