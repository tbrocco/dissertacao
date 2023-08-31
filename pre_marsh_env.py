import gymnasium
from gymnasium.utils import seeding
from gymnasium import spaces
import numpy as np
import random
from copy import deepcopy
import pygame
#import pygame_textinput
#from PIL import Image,ImageShow

GAMA = 10
STEP_REWARD = -10
MOVIMENTO_UTIL = 0.5
MOVIMENTO_INUTIL = 0
PARTIAL_WIN_GAME = 8
INVALID_MOVE = -100
INVALID_MOVE_2 = -10
INVALID_MOVE_3 = 0
WIN_GAME = 100

ENDERECO_VAZIO = -np.inf

##TODO -> Implementar o numero maximo da placa como infinito
##TODO -> Enviar a situacao inteira do patio + objetivo


def get_key(val, my_dict):
    for key, value in my_dict.items():
        if val == value:
            return key
 
    return "key doesn't exist"

class PreMarshEnv(gymnasium.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human', 'console', 'rgb_array']}

    def __init__(self,num_stacks=8,stack_height = 8, discreeteAction=True,default_occupancy=None, max_episode_steps=10, render_mode=None):
        # Define o tamanho da pilha e o número de endereços disponíveis
        self.render_mode = render_mode
        self.max_episode_steps = int(max_episode_steps)
        self.TamanhoPatio = int(num_stacks) #300
        self.TamanhoPilha = int(stack_height) #11
        
        self.pilhas_quantidade_placas = np.full(self.TamanhoPatio, -1, dtype = int)
        self.pilhas_quantidade_placas_do_objetivo = np.full(self.TamanhoPatio, -1, dtype = int)
        self.pilhas_quantidade_placas_do_objetivo_desbloqueadas = np.full(self.TamanhoPatio, -1, dtype = int)
        self.pilhas_distancia_placas_do_objetivo = np.full(self.TamanhoPatio, 0, dtype = int)
        self.quantidade_placas_do_objetivo_desbloqueadas = 0

        #self.num_areas = subYards_qty
        self.objective=[]
        self.moves = np.full(self.max_episode_steps, -1, dtype = int)
        self.lastmove = (0,0,0)
        self.dones = 0
        self.num_resets = -1
        self.total_slabs = 0
        self.reward = 0.0
        self.state = np.array([])#[self.TamanhoPatio, self.TamanhoPilha, 2]
        self.yard = []
        #self.yard_distancia = []
        #solucao 2
        self.yard_renamed = []
        self.objective_renamed = []
        self.yard_binary = []

        self.current_step = 0
        self.current_action =(0,0,0)
        # Define a taxa de ocupação padrão (50%)
        if default_occupancy is None:
            default_occupancy = random.choice([0.3, 0.5, 0.7])#random.uniform(0.3, 0.85)
        self.default_occupancy = default_occupancy
        #atualiza o total de placas suportadas no ambiente
        placas_no_ambiente = int(self.TamanhoPatio*self.TamanhoPilha*default_occupancy)
        self.total_slabs = placas_no_ambiente
        self.total_slabs_max = int(self.TamanhoPatio*self.TamanhoPilha)
        #Tamanho do Objetivo de placas no objetivo 5% do total
        self.objective_size = int(self.TamanhoPatio*self.TamanhoPilha*0.05)
        # localizacação dos objetivo
        self.localizacao = {}
        self.localizacao_posicao_pilhas = {}
        # localizacação dos objetivo
        self.objetivo_localizacao_placas = np.zeros(shape=(self.objective_size, 2), dtype = int)
        # distancia dos objetivos ao topo
        self.distancia = {}
        self.objetivo_distancia = np.zeros(shape=(self.objective_size, 2), dtype = int)
        #sample
        # self.ACTION_LOOKUP = {0 : "(0,1,0) - top slab of stack 0 to top of stack 1 ",
        #                     1 : "(0,2, 2) - three top slabs of pole 0 to top of stack 2 "}

        self.ACTION_LOOKUP = {}
        discreteAction_max = 0
        endereco = 0
        self.discreeteAction = discreeteAction
        # self.ACTION_LOOKUP[discreteAction_max] = (0,0,0) #adiciona o movimento "faz nada"
        # discreteAction_max += 1 #adiciona o movimento "faz nada" 
        while endereco < self.TamanhoPatio:
            enderecodestino =0
            while enderecodestino < self.TamanhoPatio:
                if endereco != enderecodestino:
                    for i in [0, 1, 2]:
                        self.ACTION_LOOKUP[discreteAction_max] = (endereco,enderecodestino,i)
                        discreteAction_max +=1
                enderecodestino +=1
            endereco +=1

        if discreeteAction == True :
            self.action_space = spaces.Discrete(discreteAction_max)   # lookup para o (x,y,w)


        # Define action and observation space
        
        # Define o espaço de ação
        if discreeteAction == False :
            self.action_space = spaces.Tuple((
                spaces.Discrete(self.TamanhoPatio),  # Endereço de origem da placa
                spaces.Discrete(self.TamanhoPatio),  # Endereço de destino da placa
                spaces.Discrete(3)  # Número de placas a serem movidas (1, 2 ou 3) - 0 faz nada
            ))

        # objective_space = spaces.Dict({
        #     "placas": spaces.Box(low=1, high=self.total_slabs, shape=(self.objective_size,), dtype=np.int32),
        #     "localizacao": spaces.Box(low=0,high=self.TamanhoPilha, shape=(self.objective_size, 2), dtype=np.int32),
        #     "distancia": spaces.Box(low=0,high=self.TamanhoPilha, shape=(self.objective_size,), dtype=np.int32)
        # })
        # Define o espaço de observação
        espaco_obs = {
        #'yard':  spaces.Box(low=0, high=self.total_slabs_max, shape=(2,self.TamanhoPatio, self.TamanhoPilha), dtype=np.int32), #patio self.yard + self.yard_distancia 
        #'yard':  spaces.Box(low=0, high=self.total_slabs_max, shape=(self.TamanhoPatio, self.TamanhoPilha), dtype=np.int32), #patio self.yard + self.yard_largura_placas 
        # 'objetivo' : spaces.Dict(objective_space),
        #'objetivo-placas' : spaces.Box(low=1, high=self.total_slabs, shape=(self.objective_size,), dtype=np.int32), #array objetivo
        #'objetivo-localizacao-placas' : spaces.Box(low=0, high=self.TamanhoPilha, shape=(self.objective_size,2), dtype=np.int32), #localização
        'objetivo-distancia' : spaces.Box(low=0, high=self.TamanhoPilha, shape=(self.objective_size,2), dtype=np.int32), #distancia
        'pilhas-quantidade-placas' : spaces.Box(low=0, high=self.total_slabs_max, shape=(self.TamanhoPatio,), dtype=np.int32), #pilhas_quantidade_placas
        'pilhas-quantidade-placas-objetivo' : spaces.Box(low=0, high=self.objective_size, shape=(self.TamanhoPatio,), dtype=np.int32), #pilhas_quantidade_placas_do_objetivo
        'pilhas-quantidade-placas-objetivo-desbloqueadas' : spaces.Box(low=0, high=self.objective_size, shape=(self.TamanhoPatio,), dtype=np.int32), #pilhas_quantidade_placas_do_objetivo_desbloqueadas
        'pilhas-distancia-placas-objetivo' : spaces.Box(low=0, high=self.objective_size, shape=(self.TamanhoPatio,), dtype=np.int32), #pilhas_distancia_placas_do_objetivo
        'objetivo-desbloqueadas' : spaces.Box(low=0, high=self.objective_size, shape=(1,), dtype=np.int32), #quantidade_placas_do_objetivo_desbloqueadas
        #'objetivo-localizacao' : spaces.Box(low=1, high=self.TamanhoPilha, shape=(self.objective_size,2), dtype=np.int32), #localização
        #'objetivo-distancia' : spaces.Box(low=1, high=self.TamanhoPilha, shape=(self.objective_size,), dtype=np.int32), #distancia
        'steps' : spaces.Box(low=0, high=self.max_episode_steps, shape=(1,), dtype=np.int32), #passos
        #'max_steps' : spaces.Box(low=0, high=self.max_episode_steps, shape=(1,), dtype=np.int32), #maximo de passos
        #'TamanhoPatio' : spaces.Box(low=0, high=self.TamanhoPatio, shape=(1,), dtype=np.int32), #tamanho do patio
        #'TamanhoPilha' : spaces.Box(low=0, high=self.TamanhoPilha, shape=(1,), dtype=np.int32), #tamanho da pilha
        'moves' : spaces.Box(low=-1, high=discreteAction_max, shape=(self.max_episode_steps,), dtype=np.int32), #tamanho da pilha
        }
        self.observation_space = spaces.Dict(espaco_obs)
        

        self.observation_size = self.get_space_size(self.observation_space)

        self.reward_maximo = self.objective_size*PARTIAL_WIN_GAME+self.max_episode_steps*GAMA+MOVIMENTO_UTIL+self.objective_size*100
        self.reward_range = (INVALID_MOVE, self.reward_maximo) 
        
    
    def step(self, action):
        self.current_step +=1
        truncated = False
        terminated = False
        
        # Obtém os endereços de origem e destino e o número de placas a serem movidas
        if self.discreeteAction == True:
            self.current_action = self.ACTION_LOOKUP[action] #(x,y,w)
        else:
            # Verifica se a ação é válida
            if not self.action_space.contains(action):
                self.reward = reward = INVALID_MOVE
                self.state = observation = self._get_obs()
                terminated = False
                truncated = True
                info = {}
                return observation, reward, terminated, truncated, info
            self.current_action = action
            action = get_key(action, self.ACTION_LOOKUP)
        src_stack, dst_stack, num_slabs = self.current_action
        
        num_slabs = num_slabs+1
        
        self.moves[self.current_step-1] = action
        #self.moves.append(action)

        self.lastmove = self.current_action #to be used for next validation step
        
        if num_slabs > 0:

            if self.valida(src_stack, dst_stack, num_slabs) == False:
                self.reward = reward = INVALID_MOVE
                self.state = observation = self._get_obs()
                terminated = False
                truncated = True
                info = {}
                return observation, reward, terminated, truncated, info
            
            if self.valida2(src_stack, dst_stack, num_slabs) == True:
                self.makeMove(src_stack, dst_stack, num_slabs)
                
        #atualiza o self.yard_distancia
        self.localizacao, self.distancia, self.localizacao_posicao_pilhas = self.atualizaProximidadeTopoNoObjetivo()
        self.reward = self.get_reward4(src_stack, dst_stack)

        if self.valida2(src_stack, dst_stack, num_slabs) == False:
            self.reward += INVALID_MOVE_2
        if self.valida3(src_stack, dst_stack, num_slabs) == False:
            self.reward += INVALID_MOVE_3


        win = self.verificaObjetivo3()
        
        if win:
            self.reward += WIN_GAME 
            self.dones += 1
            terminated = True
        ultimo_passo = self._check_if_maxsteps()
        if ultimo_passo:
            terminated = True
            truncated = False

        info = {}
        # info["objetivo-placas"] = self.objective
        # info["objetivo-distancia"] = self.objetivo_distancia
        
        # info["localizacao"] = self.localizacao
        # info["distancia"] = self.distancia

        # info["steps"] = self.current_step
        # info["max_steps"] = self.max_episode_steps
        # info["moves"] = self.moves

        reward = self.reward
        self.state = self._get_obs()
        observation = self.state

        return observation, reward, terminated, truncated, info
    
    # Verifica se atingiu o maximo de passos
    def _check_if_maxsteps(self):
        return self.current_step >= self.max_episode_steps
    
    def renomeiaAmbiente(self):
        objetivos_renomeados = np.arange(1, self.objective_size+1, 1, dtype=int)

        matriz_original = self.yard
        matriz_binaria = np.zeros((self.TamanhoPatio, self.TamanhoPilha), dtype=int)
        matriz_renomeada = np.zeros((self.TamanhoPatio, self.TamanhoPilha), dtype=int)
        matriz_renomeada = np.full((self.TamanhoPatio, self.TamanhoPilha), ENDERECO_VAZIO, dtype=float)
        
        for i in range(self.TamanhoPatio):
            for j in range(self.TamanhoPilha):
                if matriz_original[i][j] in self.objective:
                    index = self.objective.index(matriz_original[i][j])
                    matriz_renomeada[i][j] = objetivos_renomeados[index]
                    matriz_binaria[i][j] = 1
                elif matriz_original[i][j] == ENDERECO_VAZIO:
                     matriz_renomeada[i][j] = ENDERECO_VAZIO
                     matriz_binaria[i][j] = 0
                else:
                    matriz_renomeada[i][j] = np.inf #self.total_slabs_max+1
                    matriz_binaria[i][j] = 1
        self.yard_renamed = matriz_renomeada
        self.objective_renamed = objetivos_renomeados
        self.yard_binary = matriz_binaria

    def transformaLocalizacao(self):
        #self.objetivo_localizacao_placas =  np.array(list(self.localizacao.values()), dtype=int)
        self.objetivo_localizacao_placas =  np.array(list(self.localizacao_posicao_pilhas.values()), dtype=int)

        

    def _get_obs(self):

        self.atualizaQuantitativoPilhas()
        self.renomeiaAmbiente()
        self.transformaLocalizacao()

        spaco_obs = {
            #'yard':  np.stack([self.yard_binary, self.yard_renamed], axis=0), #patio self.yard
            #'yard':  self.yard_renamed, #patio self.yard
            # 'objetivo' : objective_space, #array objetivo
            #'objetivo-placas': self.objective_renamed,
            #'objetivo-localizacao-placas' : self.objetivo_localizacao_placas, #localização das placas
            'objetivo-distancia' : self.objetivo_distancia, #distancia
            'pilhas-quantidade-placas' : self.pilhas_quantidade_placas,
            'pilhas-quantidade-placas-objetivo' : self.pilhas_quantidade_placas_do_objetivo,
            'pilhas-quantidade-placas-objetivo-desbloqueadas' : self.pilhas_quantidade_placas_do_objetivo_desbloqueadas,
            'pilhas-distancia-placas-objetivo' : self.pilhas_distancia_placas_do_objetivo,
            'objetivo-desbloqueadas' : self.quantidade_placas_do_objetivo_desbloqueadas,
            #'objetivo-localizacao' : self.localizacao, #localização
            #'objetivo-distancia' : self.distancia, #distancia
            'steps' : self.current_step, #passos
            #'max_steps' : self.max_episode_steps, #maximo de passos
            #'TamanhoPatio' : self.TamanhoPatio, #tamanho do patio
            #'TamanhoPilha' : self.TamanhoPilha, #tamanho da pilha
            'moves' : self.moves #movimentos
        }
        self.state = spaco_obs
        
        return self.state

    def atualizaQuantitativoPilhas(self):
        objetivos = 0
        quantidade_placas = 0
        for i in range(self.TamanhoPatio):
            objetivos = 0
            quantidade_placas = 0
            for j in range(self.TamanhoPilha):
                if (self.yard[i][j] in self.objective):
                    objetivos+=1 
                if not (self.yard[i][j] == ENDERECO_VAZIO ):
                    quantidade_placas += 1   
            self.pilhas_quantidade_placas[i] = quantidade_placas
            self.pilhas_quantidade_placas_do_objetivo[i] = objetivos


    def get_space_size(self,space):
        if isinstance(space, gymnasium.spaces.Dict):
            return sum([self.get_space_size(s) for s in space.spaces.values()])
        elif isinstance(space.shape, tuple):
            return np.prod(space.shape)
        else:
            return space.shape

    # Define uma função que converte o dicionário de observação em um array unidimensional
    def dict_to_array(observation):
        return np.concatenate([
                #observation["yard"], 
                #observation["objetivo-placas"], 
                #observation["objetivo-localizacao"], 
                #observation["objetivo-localizacao-placas"], 
                observation["objetivo-distancia"],
                observation['pilhas-quantidade-placas'],
                observation['pilhas-quantidade-placas-objetivo'],
                observation['pilhas-quantidade-placas-objetivo-desbloqueadas'],
                observation['pilhas-distancia-placas-objetivo'],
                [observation['objetivo-desbloqueadas']],
                [observation["steps"]], 
                #[observation["max_steps"]], 
                #[observation["TamanhoPatio"]], 
                #[observation["TamanhoPilha"]],
                observation["moves"]
                ])
                
        #nested dict
        # return np.concatenate([observation["yard"], 
        #                 observation["objetivo"]["placas"], 
        #                 observation["objetivo"]["localizacao"], 
        #                 observation["objetivo"]["distancia"],
        #                 [observation["steps"]], 
        #                 [observation["max_steps"]], 
        #                 [observation["TamanhoPatio"]], 
        #                 [observation["TamanhoPilha"]]
        #               ])


    # Define uma função que converte o array unidimensional de observação de volta para o dicionário
    # def array_to_dict(self,obs_array):
    #     yard = obs_array[:self.TamanhoPatio*self.TamanhoPilha*2].reshape(self.TamanhoPatio, self.TamanhoPilha, 2)
    #     objective_placas = obs_array[self.TamanhoPatio*self.TamanhoPilha:self.TamanhoPatio*self.TamanhoPilha + self.objective_size]
    #     #objective_localizacao = obs_array[self.TamanhoPatio*self.TamanhoPilha + self.objective_size:self.TamanhoPatio*self.TamanhoPilha + self.objective_size + self.objective_size*2].reshape(self.objective_size, 2)
    #     #objective_distancia = obs_array[self.TamanhoPatio*self.TamanhoPilha + self.objective_size + self.objective_size*2:]
    #     steps = obs_array[self.TamanhoPatio*self.TamanhoPilha + self.objective_size + self.objective_size*2 + self.objective_size:self.TamanhoPatio*self.TamanhoPilha + self.objective_size + self.objective_size*2 + self.objective_size + 1]
    #     max_steps = obs_array[self.TamanhoPatio*self.TamanhoPilha + self.objective_size + self.objective_size*2 + self.objective_size + 1:self.TamanhoPatio*self.TamanhoPilha + self.objective_size + self.objective_size*2 + self.objective_size + 2]
    #     TamanhoPatio = obs_array[self.TamanhoPatio*self.TamanhoPilha + self.objective_size + self.objective_size*2 + self.objective_size + 2:self.TamanhoPatio*self.TamanhoPilha + self.objective_size + self.objective_size*2 + self.objective_size + 3]
    #     TamanhoPilha = obs_array[self.TamanhoPatio*self.TamanhoPilha + self.objective_size + self.objective_size*2 + self.objective_size + 3:]
    #     self.moves = obs_array[self.TamanhoPatio*self.TamanhoPilha + self.objective_size + self.objective_size*2 + self.objective_size + 3:] 
    #     return {"yard": yard, "objetivo-placas": objective_placas , 
    #             #"objetivo-localizacao": objective_localizacao ,"objetivo-distancia": objective_distancia, 
    #             "steps": steps, "max_steps": max_steps, "TamanhoPatio": TamanhoPatio, "TamanhoPilha": TamanhoPilha, "moves" : self.moves}

        #nested dict
        # yard = obs_array[:self.TamanhoPatio*self.TamanhoPilha].reshape(self.TamanhoPatio, self.TamanhoPilha)
        # objective_placas = obs_array[self.TamanhoPatio*self.TamanhoPilha:self.TamanhoPatio*self.TamanhoPilha + self.objective_size]
        # objective_localizacao = obs_array[self.TamanhoPatio*self.TamanhoPilha + self.objective_size:self.TamanhoPatio*self.TamanhoPilha + self.objective_size + self.objective_size*2].reshape(self.objective_size, 2)
        # objective_distancia = obs_array[self.TamanhoPatio*self.TamanhoPilha + self.objective_size + self.objective_size*2:]
        # objective = {"placas": objective_placas, "localizacao": objective_localizacao, "distancia": objective_distancia}
        # steps = obs_array[self.TamanhoPatio*self.TamanhoPilha + self.objective_size + self.objective_size*2 + self.objective_size:self.TamanhoPatio*self.TamanhoPilha + self.objective_size + self.objective_size*2 + self.objective_size + 1]
        # max_steps = obs_array[self.TamanhoPatio*self.TamanhoPilha + self.objective_size + self.objective_size*2 + self.objective_size + 1:self.TamanhoPatio*self.TamanhoPilha + self.objective_size + self.objective_size*2 + self.objective_size + 2]
        # TamanhoPatio = obs_array[self.TamanhoPatio*self.TamanhoPilha + self.objective_size + self.objective_size*2 + self.objective_size + 2:self.TamanhoPatio*self.TamanhoPilha + self.objective_size + self.objective_size*2 + self.objective_size + 3]
        # TamanhoPilha = obs_array[self.TamanhoPatio*self.TamanhoPilha + self.objective_size + self.objective_size*2 + self.objective_size + 3:]
        # return {"yard": yard, "objetivo": objective, "steps": steps, "max_steps": max_steps, "TamanhoPatio": TamanhoPatio, "TamanhoPilha": TamanhoPilha}


    # def __array__(self, dtype=None):
    #         return np.concatenate([self.yard_binary.reshape(-1), #self.yard_distancia.reshape(-1),
    #             self.objective,
    #             self.objetivo_distancia,
    #             [self.current_step], 
    #             [self.max_episode_steps], 
    #             [self.TamanhoPatio], 
    #             [self.TamanhoPilha],
    #             self.moves
    #             ])
    
    def stateDictToArray(self, state):
        return np.concatenate([
                #state["yard"].reshape(-1), 
                #state["yard"][0].reshape(-1),state["yard"][1].reshape(-1),
                #state["objetivo-placas"],
                # state["objetivo-distancia"],
                #state["objetivo-localizacao-placas"].reshape(-1),
                state["objetivo-distancia"].reshape(-1),
                state['pilhas-quantidade-placas'],
                state['pilhas-quantidade-placas-objetivo'],
                 state['pilhas-quantidade-placas-objetivo-desbloqueadas'],
                state['pilhas-distancia-placas-objetivo'],
                [state['objetivo-desbloqueadas']],
                [state["steps"]], 
                #[state["max_steps"]], 
                #[state["TamanhoPatio"]], 
                #[state["TamanhoPilha"]],
                state["moves"]
                ])
    

    def valida(self, src_stack, dst_stack, num_slabs):
        # Verifica se o endereço de origem é válido
        if not (0 <= src_stack < self.TamanhoPatio):
            return False
        
        # Verifica se o endereço de destino é válido
        if not (0 <= dst_stack < self.TamanhoPatio):
            return False
        
        # Verifica se o endereço de destino é válido
        if src_stack == dst_stack:
            return False 
                        
        return True
    
    def valida2(self, src_stack, dst_stack, num_slabs):
        
        # Verifica se há placas suficientes na pilha de origem
        if self.is_empty(src_stack) or (self.get_size(src_stack)) < num_slabs:
                return False
        
        # Verifica se a pilha de destino tem espaço suficiente
        if (self.TamanhoPilha - self.get_size(dst_stack)) < num_slabs:
            return False
                
        return True

    def valida3(self, src_stack, dst_stack, num_slabs):
        
        # Verifica se está realizando o movimento reverso imediatamente apos o ultimo movimento
        last_src_stack, last_dst_stack, last_num_slabs = self.lastmove
        src_stack, dst_stack, num_slabs = self.current_action
        if (src_stack == last_dst_stack and dst_stack == last_src_stack and num_slabs == last_num_slabs):
            return False
                
        return True

    # def seed(self, seed=None):
    #     self.np_random, seed = seeding.np_random(seed)
    #     return [seed]
    
    def reset(self, options=None, occupancy=None,objective=[], seed=None):
        self.stored_seed = seed
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.pilhas_quantidade_placas = np.full(self.TamanhoPatio, -1, dtype = int)
        self.pilhas_quantidade_placas_do_objetivo = np.full(self.TamanhoPatio, -1, dtype = int)
        self.pilhas_quantidade_placas_do_objetivo_desbloqueadas = np.full(self.TamanhoPatio, 0, dtype = int)
        self.pilhas_distancia_placas_do_objetivo = np.full(self.TamanhoPatio, -1, dtype = int)
        self.quantidade_placas_do_objetivo_desbloqueadas = 0
        self.reward = 0.0
        # localizacação dos objetivo
        self.objetivo_localizacao_placas = np.zeros(shape=(self.objective_size, 2), dtype = int)
        self.moves = np.full(self.max_episode_steps, -1, dtype = int)
        self.current_step = 0
        self.num_resets += 1
        self.lastmove = (0,0,0)
        # if seed is not None:
        #     self.seed = seed
        #     self.seed()
        #     random.seed(self.seed)
        # else:
        #     random.seed()

        # Define a taxa de ocupação (usa o padrão se não for especificado)
        if occupancy is None:
            occupancy = self.default_occupancy

        if occupancy is None:
            #occupancy = self.np_random.choice([0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85])#random.uniform(0.3, 0.85)
            occupancy = self.np_random.choice([0.3, 0.5, 0.7])#random.uniform(0.3, 0.85)
        
        self.default_occupancy = occupancy
        #atualiza o total de placas suportadas no ambiente
        self.total_slabs = int(self.TamanhoPatio*self.TamanhoPilha*occupancy)


        self.generate_random_map(occupancy)
        self.objetivo_distancia = np.zeros(shape=(self.objective_size ,2), dtype = int)
            
        objetivo_valido = venceu = False
        while not objetivo_valido:
            self.objective = self.defineObjetivo(objective)
            self.localizacao, self.distancia, self.localizacao_posicao_pilhas = self.atualizaProximidadeTopoNoObjetivo()
            venceu = self.verificaObjetivo3()
            #log
            # arquivo = open('arq01.txt','a')
            # texto = str(venceu) +","+  str(self.stored_seed)
            # arquivo.write('{}\n'.format(texto))
            # arquivo.close()
            objetivo_valido = not venceu         



        self.state = self._get_obs()
        info = self._get_info()
        return self.state, info

    def _get_info(self):
        return {
            "nada": 0
        }


    def close(self):
        pygame.display.quit()
        pygame.quit()

    def atualizaProximidadeTopoNoObjetivo(self):
        # Inicializa um dicionário para armazenar as distâncias relativas de cada item do objetivo em relação ao topo de cada pilha
        localizacao_patio = {}
        localizacao_posicao_pilha = {}
        distancia = {}
        arrayyard  = np.array(self.yard)
        self.pilhas_distancia_placas_do_objetivo = np.full(self.TamanhoPatio, -1, dtype = int)
        self.pilhas_quantidade_placas_do_objetivo_desbloqueadas = np.full(self.TamanhoPatio, -1, dtype = int)   

        self.atualizaQuantitativoPilhas()
        self.renomeiaAmbiente()
        index = 0
        for item in self.objective:
            indices = np.where(arrayyard == item)
            listOfCoordinates= list(zip(indices[0], indices[1]))
            for cord in listOfCoordinates:
                localizacao_patio[item] = cord
                pilhaOrigem = arrayyard[cord[0]]
                pilhaSemEspacoVazio = pilhaOrigem[pilhaOrigem!=ENDERECO_VAZIO]                
                distancia[item] = len(pilhaSemEspacoVazio)-1 - cord[1]
                if self.pilhas_distancia_placas_do_objetivo[cord[0]] == -1 :
                    self.pilhas_distancia_placas_do_objetivo[cord[0]] = 0
            
                self.pilhas_distancia_placas_do_objetivo[cord[0]] += distancia[item]

                indices_posicao_pilha = np.where(pilhaSemEspacoVazio == item)
                localizacao_posicao_pilha[item] = (cord[0], int(indices_posicao_pilha[0]))
                
            index +=1        

        self.quantidade_placas_do_objetivo_desbloqueadas = 0 
        index = 0
        for item in self.objective:
            i_pilha = localizacao_patio[item][0]
            i_posicao_placa = localizacao_patio[item][1]
            #atualiza os vetores
            if self.pilhas_quantidade_placas_do_objetivo_desbloqueadas[i_pilha] == -1 :
                self.pilhas_quantidade_placas_do_objetivo_desbloqueadas[i_pilha] = 0
            #verifica desbloqueio
            if (distancia[item] == int(0)):
                self.quantidade_placas_do_objetivo_desbloqueadas += 1
                self.pilhas_quantidade_placas_do_objetivo_desbloqueadas[i_pilha] += 1 
            else:
                for index_v in range(i_posicao_placa,self.pilhas_quantidade_placas[i_pilha]):
                    if self.yard_renamed[i_pilha][i_posicao_placa] > self.yard_renamed[i_pilha][index_v]:
                        distancia[item] -= 1
                        self.pilhas_distancia_placas_do_objetivo[i_pilha] += -1
                        if (distancia[item] == int(0)):
                            self.quantidade_placas_do_objetivo_desbloqueadas += 1
            #atualiza o vetor com o mesmo index do dict
            self.objetivo_distancia[index] = [i_pilha, distancia[item]]
            index +=1  

        return localizacao_patio, distancia, localizacao_posicao_pilha

    def get_reward4(self, src_stack, dst_stack):
        recompensa_total_objetivo = 0
        reward_movimento_util = 0
        recompensa_descontada = (self.max_episode_steps - self.current_step)
        arraystate = np.array(self.yard)

        sourceSlabStack = arraystate[src_stack]
        destSlabStack = arraystate[dst_stack]

        sourceIntersection = np.intersect1d(self.objective, sourceSlabStack)
        destIntersection = np.intersect1d(self.objective, destSlabStack)

        if (len(sourceIntersection) + len(destIntersection)) < 0:
            reward_movimento_util = MOVIMENTO_INUTIL
        else:
            reward_movimento_util = MOVIMENTO_UTIL

        # Inicializa recompensa total
        reward_final = 0
    
        for item in self.objective:
            indices = np.where(arraystate == item)
            listOfCoordinates = list(zip(indices[0], indices[1]))

            # Inicializa recompensas específicas para cada objetivo
            reward_distancia = 0
            recompensa_completude = 0
            
            for cord in listOfCoordinates:
                reward_distancia += self.TamanhoPilha - self.distancia[item]
            
            valor_maximo = self.TamanhoPilha * len(self.objective)
            porcentagem_completude = (reward_distancia / valor_maximo) * 100
            recompensa_completude = porcentagem_completude
            
            # Calcula recompensa total para o objetivo atual
            #recompensa_total_objetivo = self.quantidade_placas_do_objetivo_desbloqueadas * GAMA
            recompensa_total_objetivo += recompensa_completude

        recompensa_total_objetivo += reward_movimento_util
        recompensa_total_objetivo += self.current_step * STEP_REWARD            
        
        recompensa_sequencia = 0

        for idx in range(self.quantidade_placas_do_objetivo_desbloqueadas):
            item = self.objective[idx]
            if self.distancia[item] == 0:
                recompensa_sequencia += PARTIAL_WIN_GAME

        recompensa_total_objetivo += recompensa_sequencia

        reward_final = recompensa_total_objetivo

        return reward_final
        
    def defineObjetivo(self, objective):
        if len(objective) == 0:
            objective = []
            #patio_plano = self.yard.flatten()
            #self.np_random.shuffle(patio_plano)
            ids = self.np_random.choice(range(1,self.total_slabs), 3, replace=False)
            #ids = random.sample(range(1, self.total_slabs), self.objective_size)
            for i in range(self.objective_size):
                slab = int(ids[i])
                objective.append(slab)
        # else:
        #     self.objective_size = len(objective)

        return objective

    def generate_random_map(self, occupancy):
        self.yard = np.zeros((self.TamanhoPatio, self.TamanhoPilha), dtype=np.int32)
        self.yard = np.full((self.TamanhoPatio, self.TamanhoPilha), ENDERECO_VAZIO, dtype=float)
        
        #self.yard_distancia = np.zeros((self.TamanhoPatio, self.TamanhoPilha), dtype=np.int32)
        for i in range(1, self.total_slabs+1):
            slab = i
            incluido = True
            while True:
                if (self.add_slab(self.np_random.integers(0, self.TamanhoPatio-1), slab)):
                    break
                else:
                    incluido = False
                    

    def verificaObjetivo(self):
        # cria uma cópia profunda do ambiente
        ambiente_copia = deepcopy(self.yard)
        found = 0
        try:
            # Verifica se o objetivo foi alcançado
            for slab in enumerate(self.objective):
                for slabStack in enumerate(self.yard):
                    if not self.is_empty(slabStack[0]) and (self.is_slab_on_top(slabStack[1], slab[1])):
                        found = found + 1
                        self.remove_slab(slabStack[0])
                        break
        except:
            raise ("erro")
        finally:
            self.yard = deepcopy(ambiente_copia)       
        if found == len(self.objective):
            return True, found
        return False, found
    
    def verificaObjetivo2(self):
        resultado = self.yard_distancia[np.nonzero(self.yard_distancia)]
        if len(resultado) > 0 :
            return False, len(resultado)
        else:
            return True, len(resultado)
        
    def verificaObjetivo3(self):
        #resultado = np.where(self.yard_distancia==int(1))
        # resultado = np.where(self.objetivo_distancia==int(1))
        # listOfCoordinates= list(zip(resultado[0], resultado[1]))
        all_ones = np.all(self.objetivo_distancia[:, 1] == int(0))
        if all_ones :
            return True
        else:
            return False
    
        

    def remove_slab(self,address_stack):
        pilha = self.yard[address_stack]
        pilhaSemEspacoVazio = pilha[pilha!=ENDERECO_VAZIO]
        if len(pilhaSemEspacoVazio) > 0:
            topo = len(pilhaSemEspacoVazio)-1
            slab = self.yard[address_stack][topo]
            self.yard[address_stack][topo] = ENDERECO_VAZIO
            return slab
        else:
            #raise Exception("No slab found at address {}".format(address))
            return False
        
    def add_slab(self,address_stack, slab):
        pilha = self.yard[address_stack]
        pilhaSemEspacoVazio = pilha[pilha!=ENDERECO_VAZIO]
        if not self.is_full(address_stack):
            if (len(pilhaSemEspacoVazio) == 0):
                pilha[0] = slab #coloca a placa do topo
            else:
                i = 1
                while i < len(pilha):
                    if pilha[i] == ENDERECO_VAZIO:
                        pilha[i] = slab
                        break
                    i += 1
            
            self.yard[address_stack] = pilha #devolve a pilha modificada
            return True
        else:
            #raise Exception("Stack is full at address {}".format(address))
            return False

    def get_size(self,address_stack):
        pilha = self.yard[address_stack]
        pilhaSemEspacoVazio = pilha[pilha!=ENDERECO_VAZIO]
        return len(pilhaSemEspacoVazio)

    def is_full(self,address_stack):
        pilha = self.yard[address_stack]
        pilhaSemEspacoVazio = pilha[pilha!=ENDERECO_VAZIO]
        return len(pilhaSemEspacoVazio) == self.TamanhoPilha

    def is_empty(self,address_stack):
        pilha = self.yard[address_stack]
        pilhaSemEspacoVazio = pilha[pilha!=ENDERECO_VAZIO]
        return len(pilhaSemEspacoVazio) == 0
    

    def is_slab_on_top(self,stack, slab_id):
        pilha = stack
        pilhaSemEspacoVazio = pilha[pilha!=ENDERECO_VAZIO]
        # Verifica se a pilha está vazia
        if pilhaSemEspacoVazio[-1] == 0:
            return False
        
        # Verifica se a placa está no topo da pilha
        return pilhaSemEspacoVazio[-1] == slab_id
    
    # Function to apply a move to a state and return the new state.
    def makeMove(self, src_stack, dst_stack, num_slabs):
        buffer_slab = []
        try:
            for _ in range(num_slabs):
                slab = self.remove_slab(src_stack)
                buffer_slab.append(slab)

            for _ in range(num_slabs):
                self.add_slab(dst_stack, buffer_slab.pop())
        except:
            return False
        return True

    def render(self):    
        if (self.render_mode=='console'):
            # print("Resets: ", self.num_resets)
            # print("Objetivo: ", self.objective)
            # print("Reward: ", self.reward)
            # print("Step: ", self.current_step)
            # print("Action: ", self.current_action)
            # print("Moves: ", self.current_action)
            # print("Dones: ", self.dones)
            print(self.state)

            # #print(self.yard) 
            # for i in range(self.TamanhoPatio):
            #     stack_str = ''
            #     stack_str += str([slab for slab in self.yard[i]]) + ' | '
            #     print(f'Stack {i}: {stack_str}')

            # #print(self.yard) 
            # for i in range(self.TamanhoPatio):
            #     stack_str = ''
            #     stack_str += str([slab for slab in self.yard_distancia[i]]) + ' | '
            #     print(f'Distance {i}: {stack_str}')
        else:
            
            # Define RENDER
            square_size = 20
            margin = 5
            marginleft = 15
            toolbar = 40
            footer = 30
            colors = {
                'empty': (255, 255, 255),
                'slab': (0, 0, 255),
                'robot': (255, 0, 0),
                'done': (255, 0, 255),
            }


            num_stacks = self.TamanhoPatio
            stack_height = self.TamanhoPilha
            screen_width = num_stacks * (square_size + margin) + margin + toolbar
            screen_height = (stack_height + 1) * (square_size + margin) + margin + toolbar + footer

            pygame.init()
            pygame.font.init()

            screen = pygame.display.set_mode((screen_width, screen_height))
            pygame.display.set_caption("Yard Pre Marsh")

            # Cria um objeto de fonte
            font = pygame.font.SysFont(None, 18)
            text_color = (0, 255, 0)
            text_obj_color  = (255, 0, 0)
            fim = None
            clock = pygame.time.Clock()


            #Criar CAIXA DE TEXT
            # text_input = pygame_textinput.TextInputVisualizer()
            # text_input.font_color = (255, 255, 10)
            #FIM

            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                    if self.render_mode == 'human':
                        
                        # checking if keydown event happened or not
                        if event.type == pygame.KEYDOWN:
                            # if event.unicode.isprintable():
                            #    text_input.value += event.unicode
                            # elif event.key == pygame.K_RETURN:
                            #     input_text = text_input.value  # Obtém o texto digitado pelo usuário
                            #     text_input.value = ""  # Limpa a caixa de texto
                            #     # Realiza a ação com o texto de entrada
                            #     src_stack, dst_stack, num_slabs = input_text.split()
                            #     action = int(src_stack), int(dst_stack), int(num_slabs) - 1
                            #     observation, reward, done, info = self.step(action)  # Execute a ação e receba o resultado
                            #     if done:
                            #         print("Done!!!")
                            #         fim = done
                            if event.key == pygame.K_m:
                                src_stack, dst_stack, num_slabs = input("Entre com o endereço de origem, destino, e quantidade de placas ").split()
                                action = int(src_stack), int(dst_stack), int(num_slabs)-1
                                observation, reward, terminated, truncated, info = self.step(action)  # Execute a ação e receba o resultado
                                done = terminated + truncated
                                print(f'Reward {reward}')
                                print(self.state)
                                if done:
                                    print("Done!!!")
                                    fim = done
                                    
                                # get text input from 0 to -1 i.e. end.
                                #user_text = user_text[:-1]
                
                            # Unicode standard is used for string
                            # formation
                            #else:
                                #user_text += event.unicode

                screen.fill((0, 0, 0))


                for stack_index in range(num_stacks):
                    stack = self.yard[stack_index]
                    
                    #label de coluna
                    x = marginleft + stack_index * (square_size + margin) + square_size // 2
                    y = (self.TamanhoPilha + 1) * (square_size + margin) + margin + toolbar
                    label_text = font.render(str(stack_index), True, text_color)
                    label_rect = label_text.get_rect(center=(x, y))
                    screen.blit(label_text, label_rect)


                    for slab_index in range(len(stack)):
                        slab = stack[slab_index]

                        if float(slab) > ENDERECO_VAZIO:
                            color = colors['slab']
                            x = marginleft + stack_index * (square_size + margin)
                            y = (self.TamanhoPilha - slab_index) * (square_size + margin) + toolbar
                            rect = pygame.draw.rect(screen, color, (x, y, square_size, square_size))
                            if any(x == slab for x in self.objective) :
                                text = font.render(str(slab), True, text_obj_color)
                            else:
                                text = font.render(str(slab), True, text_color)
                            text_rect = text.get_rect(center=rect.center)
                            screen.blit(text, text_rect)

                for row_index in range(self.TamanhoPilha):
                    #label de linha
                    x = marginleft // 2
                    y = (self.TamanhoPilha - row_index) * (square_size + margin) + square_size // 2 + toolbar
                    label_text = font.render(str(row_index), True, text_color)
                    label_rect = label_text.get_rect(center=(x, y))
                    screen.blit(label_text, label_rect)

                    # for i in range(len(stack), self.TamanhoPilha):
                    #     color = colors['empty']
                    #     x = margin + stack_index * (square_size + margin)
                    #     y = margin + (self.TamanhoPilha - i - 1) * (square_size + margin)
                    #     pygame.draw.rect(screen, color, (x, y, square_size, square_size))
                if fim:
                    color = colors['empty']
                    text_obj_color = colors['empty']
                    x = screen.get_width()/2
                    y = screen.get_height()/2
                    rect = pygame.draw.rect(screen, color, (x, y, 200, 200))
                    text = font.render(str("FIM!!!! PARABENS"), True, text_color)
                    text_rect = text.get_rect(center=rect.center)
                    screen.blit(text, text_rect)
                
                # text_input.update(pygame.event.get())
                # # Desenha a caixa de texto na tela
                texto_explicativo = "SEED:"
                texto_explicativo += str(self.stored_seed)
                texto_explicativo += "Objetivo:"
                texto_explicativo += str(self.objective)
                text = font.render(str(texto_explicativo), True, text_color)
                text_rect = text.get_rect(center=(100,5))
                screen.blit(text, text_rect)
                texto_explicativo = "Step: "
                texto_explicativo += str(self.current_step)
                texto_explicativo += "Move: "
                texto_explicativo += str(self.current_action)
                text = font.render(str(texto_explicativo), True, text_color)
                text_rect = text.get_rect(center=(100,20))
                screen.blit(text, text_rect)
                texto_explicativo = "Reward:"
                texto_explicativo += str(self.reward)
                text = font.render(str(texto_explicativo), True, text_color)
                text_rect = text.get_rect(center=(100,35))
                screen.blit(text, text_rect)
                pygame.display.flip()
                pygame.display.update()
                clock.tick(30)
                if self.render_mode == "rgb_array":
                    array = pygame.surfarray.array3d(screen)
                    array = np.flip(array, axis=1)
                    return np.rot90(array, k=1)
                
        return None
