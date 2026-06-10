from functools import staticmethod
from dataclasses import dataclass
from engine.environment.state import GameState
from typing import Dict, List
import numpy as np

@dataclass
class ObservationFormat:
    self_features : Dict 
    ally_features : List[Dict]
    enemy_features : List[Dict]

    def to_vector(self) -> np.ndarray:
        vector = []

        vector.extend([
            self.self_features['hp'],
            self.self_features['x'],
            self.self_features['y'],
            self.self_features['skill_1_ready'],
            self.self_features['skill_2_ready'],
         ])

        for ally in self.ally_features:

            vector.extend([
                ally['hp'],
                ally['x'],
                ally['y'],
                ally['skill_1_ready'],
                ally['skill_2_ready'],
            ])
        
        for enemy in self.enemy_features:

            vector.extend([
                enemy['hp'],
                enemy['x'],
                enemy['y'],
            ])

        self.array = np.array(vector , dtype = np.float32)
        
        return self.array

    @staticmethod
    def vector_size(array):
        return len(array)

class ObservationEncoder:

    VISION_RANGE = 3

    @staticmethod
    def build_observation(agent_id : int, gamestate : GameState):

        role = gamestate.identities[agent_id].role
        team = gamestate.teams[agent_id]

        self_features = ObservationEncoder._extract_agent_features(agent_id, gamestate)

        ally_features = []
        enemy_features = []

        if team == 'heroes':

            hero_roles = {
                gamestate.identities[aid].role : aid for aid, t in gamestate.teams.items() if t == 'heroes'
            }

            if role == 'Tank':
                ally_order = [
                    hero_roles.get('Dealer'),
                    hero_roles.get('Healer')
                ]
            
            elif role == 'Dealer':
                ally_order = [
                    hero_roles.get('Tank'),
                    hero_roles.get('Healer')
                ]
            
            else :
                ally_order = [   ## Decide the order of the allies I have to append in the ally features
                    hero_roles.get('Tank'),
                    hero_roles.get('Dealer')
                ]
            

            for aid in ally_order:   #For allies.. we use the same helper function which we used for self features. Simply give a id and gamestate and it extracts

                ally_features.append(
                    ObservationEncoder._extract_agent_features(aid, gamestate)
                )
            
            boss_id = next((bid for bid, t in gamestate.teams.items() if t =='boss'),None)   ## Next method creates a geenrato style iterator here. We return the very first time we get some boss, as we have only one boss
        
            boss_visible = False

            if gamestate.is_alive(boss_id):
                for hid , t in gamestate.teams.items():
                    if t == 'heroes' and gamestate.is_alive(hid):
                        if gamestate.distance(hid , boss_id) <= ObservationEncoder.VISION_RANGE:
                            boss_visible = True
                            break
                        
                    
            if boss_visible:
                enemy_features.append(
                    ObservationEncoder._extract_enemy_features(
                        boss_id, gamestate
                    )
                )
            else :
                enemy_features.append({
                    'hp' : 0.0,
                    'x'  : -1.0,
                    'y'  : -1.0
                })
            
        else : ## If team is boss, that is if we are building boss's perspective

            hero_ids = [
                aid for aid, t in gamestate.teams.items() if t == 'heroes'
            ]

            for hid in hero_ids:
                visible = (
                    gamestate.is_alive(hid) and gamestate.distance(agent_id, hid) <= ObservationEncoder.VISION_RANGE
                )
                
                if visible:
                    enemy_features.append(
                        ObservationEncoder._extract_enemy_features(hid, gamestate)
                    )
                
                else:                
                    enemy_features.append({
                        'hp' : 0.0,
                        'x'  : -1.0,
                        'y'  : -1.0
                    })
                
        return ObservationFormat(
            self_features = self_features,
            ally_features = ally_features,
            enemy_features = enemy_features,
        )

    @staticmethod
    def _extract_agent_features(agent_id : int, gamestate : GameState) -> Dict:

        if agent_id is None or not gamestate.is_alive(agent_id):
            return {
                'hp' : 0.0,
                'x' : -1.0,
                'y' : -1.0,
                'skill_1_ready' : 0,
                'skill_2_ready' : 0,
            }

        pos = gamestate.positions[agent_id]
        cools = gamestate.cooldowns.get(agent_id, [0,0])

        return {
            'hp' : gamestate._hp_ratio(agent_id),
            'x' : pos[0]/gamestate.grid_size ,
            'y' : pos[1]/gamestate.grid_size,
            'skill_1_ready' : 1.0 if cools[0] == 0.0 else 0.0,
            'skill_2_ready' : 1.0 if cools[1] == 0.0 else 0.0,
        }

    
    @staticmethod
    def _extract_enemy_features(enemy_id : int, gamestate : GameState) -> Dict:

        if not gamestate.is_alive(enemy_id):
            return {
                "hp": 0.0,
                "x": -1.0,
                "y": -1.0,
            }

        pos = gamestate.positions[enemy_id]

        return {
            "hp" : gamestate._hp_ratio(enemy_id),
            "x" : pos[0] / gamestate.grid_size,
            "y" : pos[1] / gamestate.grid_size,
        }