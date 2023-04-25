import gym
import copy
from gym import spaces
import numpy as np
import random
import pygame
from PIL import Image,ImageShow


# from PIL import Image, ImageDraw
# import matplotlib.pyplot as plt

class Slab:
    def __init__(self, slab_id, slab_type=None, size=1, area=None, address=None):
        self.slab_id = slab_id
        self.slab_type = slab_type
        self.size = size
        self.area = area
        self.address = address

class SlabStack:
    def __init__(self, stack_height=11):
        self.stack_height = stack_height
        self.stack = []
        # self.id = id
        # self.area = area
        # self.endereco = f'{area}{id}'

    def __array__(self, dtype=None):
        array = []
        for i in range(len(self.stack)):
            array.append(self.stack[i].slab_id) 
        return np.array(array)

    def is_full(self):
        return len(self.stack) == self.stack_height
   
    def is_empty(self):
        return len(self.stack) == 0
    
    def get_size(self):
        return len(self.stack)

    def add_slab(self, slab):
        if not self.is_full():
            self.stack.append(slab)
            return True
        else:
            return False
            #raise Exception("Stack is full at address {}".format(address))

    def remove_slab(self):
        if len(self.stack) > 0:
            return self.stack.pop()
        else:
            #raise Exception("No slab found at address {}".format(address))
            return False

    def get_top_slabs(self, num_slabs):
        if len(self.stack) >= num_slabs:
            return self.stack[-num_slabs:]
        else:
            # raise Exception("Not enough slabs at address {}".format(address))
            raise Exception("Not enough slabs at address")
    
    def get_slab_by_index(self, index):
        if len(self.stack) > 0 and len(self.stack) >= index:
            return self.stack[index] 
        else:
            raise Exception("Invalid Index at STACK")
    
class Yard:
    def __init__(self, num_stacks,stack_height):
        self.num_stacks = num_stacks
        self.stack_height = stack_height
        self.stacks = []
        pilhas = []
        for i in range(int(num_stacks)):
                pilhas.append(SlabStack(stack_height))
        self.stacks = pilhas

    def __array__(self, dtype=None):
        array = []
        for i in range(self.num_stacks):
            arrayInner = []
            #for j in range (len(self.areas[i // 100][i % 100].stack)):\
            for j in range (self.stack_height):
                if j < len(self.stacks[i].stack):
                    arrayInner.append(self.stacks[i].stack[j].slab_id)
                else:
                    arrayInner.append(0)
            array.append(arrayInner)
        return np.array(array)

    def add_slab(self, slab, address):
        #return self.areas[address // 100][address % 100].add_slab(slab)
        return self.stacks[address].add_slab(slab)

    def remove_slab(self, address):
        #return self.areas[address // 100][address % 100].remove_slab()
        return self.stacks[address].remove_slab()
    

    def get_top_slabs(self, address, num_slabs):
        #return self.areas[address // 100][address % 100].get_top_slabs(num_slabs)
        return self.stacks[address].get_top_slabs(num_slabs)
    
    def is_full(self, address):
        #self.areas[address // 100][address % 100].is_full()
        self.stacks[address].is_full()

    def is_empty(self, address):
        #self.areas[address // 100][address % 100].is_empty()
        self.stacks[address].is_empty()
    
class YardEnv(gym.Env):
    def __init__(self,num_stacks=10,stack_height = 11, discreeteAction=False,seed=1, objective=[]):
        #self.subYards = [Yard(100) for _ in range(3)]
        self.yard = Yard(num_stacks, stack_height)
        self.robot_positions = [0 for _ in range(2)] #origem e destino
        
        #self.map_code = self.generate_map_code(map_code)
       
        # Define o tamanho da pilha e o número de endereços disponíveis
        self.stack_height = stack_height
        self.num_stacks = num_stacks
        #self.num_areas = subYards_qty
        self.objective=objective
        self.moves = []
        self.lastmove = (0,0)
        self.cost = 0
        self.num_steps = 0
        self.num_resets = -1
        self.p_reward = 0 
        self.t_reward = 0
        self.total_slabs = 0
        self.seed = seed
        # Define o espaço de ação
        if discreeteAction == False :
            self.action_space = spaces.Tuple((
                spaces.Discrete(self.num_stacks),  # Endereço de origem da placa
                spaces.Discrete(self.num_stacks),  # Endereço de destino da placa
                spaces.Discrete(3)  # Número de placas a serem movidas (1, 2 ou 3)
            ))
        else: 
            self.action_space = spaces.Discrete(self.num_stacks*(self.num_stacks-1))
        # Define o espaço de observação
        self.observation_space = spaces.Box(low=0, high=self.stack_height, shape=(self.num_stacks, self.stack_height), dtype=np.int32)
        
        self.ACTION_LOOKUP = {0 : "(0,1) - top disk of pole 0 to top of pole 1 ",
                              1 : "(0,2) - top disk of pole 0 to top of pole 2 ",
                              2 : "(1,0) - top disk of pole 1 to top of pole 0",
                              3 : "(1,2) - top disk of pole 1 to top of pole 2",
                              4 : "(2,0) - top disk of pole 2 to top of pole 0",
                              5 : "(2,1) - top disk of pole 2 to top of pole 1"}

        self.ACTION_LOOKUP = {}
        discreteAction = 0
        endereco = 0
        self.discreeteAction = discreeteAction
        if discreeteAction == True :
            while endereco < self.num_stacks:
                enderecodestino =0
                while enderecodestino < self.num_stacks:
                    if endereco != enderecodestino:
                        self.ACTION_LOOKUP[discreteAction] = [endereco,enderecodestino,0]
                        discreteAction +=1
                    enderecodestino +=1
                endereco +=1

            self.action_space = spaces.Discrete(discreteAction)   # lookup para o (x,y)

        # Define a taxa de ocupação padrão (50%)
        self.default_occupancy = 0.5
        
        # Define a matriz de estado inicial (todas as pilhas vazias)
        #self.state = np.zeros((self.num_stacks, self.stack_size), dtype=np.int32)
        self.state = self.yard

    def step(self, action):
        self.num_steps = self.num_steps + 1
        INVALID_MOVE = -1000
   
        # Obtém os endereços de origem e destino e o número de placas a serem movidas
        if self.discreeteAction == True:
            action = self.ACTION_LOOKUP[action] #(x,y)
        else:
            # Verifica se a ação é válida
            if not self.action_space.contains(action):
                self.p_reward = INVALID_MOVE
                return self.state, self.p_reward, False, {}
        src_stack, dst_stack, num_slabs = action
        
        num_slabs = num_slabs+1
        
        self.moves.append(action)
        self.lastmove = action
        #captura de qual subpátio deve ser feita a transferencia (ou o endereço pertence)
        #src_subYards = src_stack // 100
        #dst_subYards = dst_stack // 100

        # Verifica se o endereço de origem é válido
        if not (0 <= src_stack < self.num_stacks):
            self.p_reward = INVALID_MOVE
            return self.state, self.p_reward, False, {}
        
        # Verifica se o endereço de destino é válido
        if not (0 <= dst_stack < self.num_stacks):
            self.p_reward = INVALID_MOVE
            return self.state, self.p_reward, False, {}
        
        # Verifica se há placas suficientes na pilha de origem
        #if self.state.areas[dst_subYards][src_stack].is_empty() or (self.state.areas[dst_subYards][src_stack].get_size()) < num_slabs:
        if self.state.stacks[src_stack].is_empty() or (self.state.stacks[src_stack].get_size()) < num_slabs:
              self.p_reward = INVALID_MOVE
              return self.state, self.p_reward, False, {}
        
        # Verifica se a pilha de destino tem espaço suficiente
        #if (self.stack_height - self.state.areas[dst_subYards][dst_stack].get_size()) < num_slabs:
        if (self.stack_height - self.state.stacks[dst_stack].get_size()) < num_slabs:
            self.p_reward = INVALID_MOVE
            return self.state, self.p_reward, False, {}
        
        buffer_slab = []
        for _ in range(num_slabs):
            slab = self.state.remove_slab(src_stack)
            buffer_slab.append(slab)

        for _ in range(num_slabs):
            self.state.add_slab(buffer_slab.pop(), dst_stack)
            
        # if src_subYards == dst_subYards:
        #     self.cost += -1
        # else:
        #     self.cost += -5
        self.cost += 1

        #move o robo
        self.robot_positions[0] = src_stack
        self.robot_positions[1] = dst_stack


        # # Move as placas da pilha de origem para a pilha de destino
        # self.state[dst_stack][num_slabs:num_slabs*2] = self.state[src_stack][:num_slabs]
        # self.state[src_stack][:num_slabs] = 0
        
        # Calcula a recompensa 
        # (0.1 ponto por placa da pilha movida)
        # (0.5 pontos por placa desbloqueada na pilha)
        # (11 pontos caso a pilha esteja resolvida )

        reward = self.p_reward = self.get_reward(src_stack, dst_stack)
        
        # Verifica se o jogo acabou (placas desbloqueadas)
        done, self.t_reward = self.verificaObjetivo()
        if done:
            reward = self.t_reward
        
        self.yard = self.state

        return np.array(self.state), reward, done, {}

    def reset(self,seed=None, occupancy=None,objective=[]):
        # Reinicia a matriz de estado (todas as pilhas vazias)
        # self.state = np.zeros((self.num_stacks, self.stack_size), dtype=np.int32)
        # return self.state
        self.moves = []
        self.cost = 0
        self.num_steps = 0
        
        self.num_resets =self.num_resets + 1
        if seed is not None:
            self.seed = seed 
            random.seed(self.seed)
        else:
            random.seed()

            # Define a taxa de ocupação (usa o padrão se não for especificado)
        if occupancy is None:
            occupancy = self.default_occupancy
        
        # Gera um pátio aleatório com base na taxa de ocupação
        #stacks = [self.generate_random_stack(occupancy) for _ in range(self.num_stacks)]
        
        # Define o estado inicial com as pilhas aleatórias
        #self.state = np.array(stacks)
        
        self.state = self.generate_random_map(occupancy)
        self.objective = self.defineObjetivo(objective)
        self.yard = self.state
        return np.array(self.state)

    def get_reward(self, src_stack, dst_stack):
        # Inicializa um dicionário para armazenar as distâncias relativas de cada item do objetivo em relação ao topo de cada pilha
        localizacao = {}
        distancia = {}
        ENDERECO_VAZIO = 0
        soma_da_proximidade_do_topo_dos_objetivos = 0
        arraystate  = np.array(self.state)

        sourceSlabStack = arraystate[src_stack]
        destSlabStack = arraystate[dst_stack]

        # Usa a função intersect1d para encontrar os elementos comuns aos dois arrays
        sourceIntersection = np.intersect1d(self.objective, sourceSlabStack)
        destIntersection = np.intersect1d(self.objective, destSlabStack)

        if (len(sourceIntersection) + len(destIntersection) )< 0:
            return -1
        else:


            for item in self.objective:
                indices = np.where(arraystate == item)
                listOfCoordinates= list(zip(indices[0], indices[1]))
                for cord in listOfCoordinates:
                    localizacao[item] = cord
                    pilhaOrigem = arraystate[cord[0]]
                    pilhaSemEspacoVazio = pilhaOrigem[pilhaOrigem!=ENDERECO_VAZIO]                
                    distancia[item] = len(pilhaSemEspacoVazio) - cord[1]
                    soma_da_proximidade_do_topo_dos_objetivos += (self.stack_height - distancia[item])

            return soma_da_proximidade_do_topo_dos_objetivos


    def comentario(self, src_stack, dst_stack):
        return False
        # Calcula a recompensa 
        # (1 ponto pela quantidade de placas existente na pilha de origem)
        # (0.1 ponto pela quantidade de placas existente na pilha de destino)
        # (0.5 pontos por placa desbloqueada na pilha)
        # (11 pontos caso a pilha esteja resolvida )
        #partialgoal = len(self.objective)
        # partialReward = 0
        # achievedgoal = []
        # # cria uma cópia profunda do ambiente
        # ambiente_copia = copy.deepcopy(self.state)
        # found = 0
        # i = 0
        # nparraystate = np.array(self.state)
        # while i < len(self.objective):
        #     distancia = self.calcula_distancia_topo(nparraystate, self.objective[i])
        #     partialReward = partialReward + distancia
        #     #incluir a verificacao de quem esta em cima 
        #     i+=1
        # return partialReward 


        # sourceSlabStack = ambiente_copia.stacks[src_stack]
        # destSlabStack = ambiente_copia.stacks[dst_stack]

        # # Usa a função intersect1d para encontrar os elementos comuns aos dois arrays
        # sourceIntersection = np.intersect1d(self.objective, np.array(sourceSlabStack))
        # destIntersection = np.intersect1d(self.objective, np.array(destSlabStack))

        # partialReward = len(sourceIntersection)*1 + len(destIntersection)*0.1
        
        # i = 0
        # while i < len(self.objective):
        #     if sourceSlabStack.get_size() > 0  and sourceSlabStack.get_top_slabs(1)[-1].slab_id == self.objective[i]:
        #     # if sourceSlabStack[-1].slab_id == self.objective[i]:
        #         achievedgoal.append(sourceSlabStack.remove_slab())
        #     i += 1
        # i = 0
        # while i < len(self.objective):
        #     if destSlabStack.get_size() >0  and destSlabStack.get_top_slabs(1)[-1].slab_id == self.objective[i]:
        #         achievedgoal.append(destSlabStack.remove_slab())
        #     i += 1

        # partialReward = partialReward + len(achievedgoal)*3
        # return partialReward 

    def encontra_item(self, matriz, item):
        for i in range(len(matriz)):
            for j in range(len(matriz[i])):
                if matriz[i][j] == item:
                    # Item encontrado, retorna sua posição
                    return (i, j)
        # Item não encontrado, retorna None
        return None

    def calcula_distancia_topo(self, matriz, item):
        posicao = self.encontra_item(matriz, item)
        if posicao is None:
            # Item não encontrado, retorna None
            return None
        # Calcula a distância do item até o topo da pilha
        pilha = matriz[posicao[0]]
        pilhaSemZero = pilha[pilha!=0]
        distancia = len(pilhaSemZero) - 1 - posicao[1]
        return distancia   


    def get_cost(self):
        return self.cost
    
    def get_moves(self):
        return self.moves

    def defineObjetivo(self, objective):
        if len(objective) == 0:
            num_slabs = int(self.num_stacks*self.stack_height*0.05)
            objective = []
            ids = random.sample(range(1, self.total_slabs), num_slabs)
            for i in range(num_slabs):
                slab = Slab(int(ids[i]) )
                objective.append(slab.slab_id)

        return objective

    def generate_random_map(self, occupancy):
        temp = Yard(self.num_stacks, self.stack_height)
        
        self.total_slabs = int(self.num_stacks*self.stack_height*occupancy)

        #slabs_ids = random.sample(range(1, total+1), total+1)        
        for i in range(1, self.total_slabs+1):
            slab = Slab(i)

            while temp.add_slab(slab,random.randint(0, self.num_stacks-1)):
                break
        
        return temp

    def verificaObjetivo(self):
        # cria uma cópia profunda do ambiente
        ambiente_copia = copy.deepcopy(self.state)
        found = 0
        # Exemplo de array objetivo com sequência de placas
        #objetivo = [3, 7, 5, 2]
         # Verifica se o objetivo foi alcançado
        for slab in enumerate(self.objective):
            for slabStack in enumerate(ambiente_copia.stacks):
                if len(slabStack[1].stack) > 0 and (self.is_slab_on_top(slabStack[1].stack, slab[1])):
                    found = found + 1
                    ambiente_copia.stacks[slabStack[0]].remove_slab()
                    break
        if found == len(self.objective):
            return True, found
        return False, found
    
    def is_slab_on_top(self,stack, slab_id):
        # Verifica se a pilha está vazia
        if stack[-1] == 0:
            return False
        
        # Verifica se a placa está no topo da pilha
        return stack[-1].slab_id == slab_id

    def render(self, mode='rgb_array'):
        if mode == 'console':
            # Imprime o estado atual do pátio no console
            for i in range(self.num_stacks):
                stack_str = ''
                stack_str += str([slab.slab_id for slab in self.yard.stacks[i].stack]) + ' | '
                print(f'Stack {i}: {stack_str}')
            print(f'Ultimo Movimento: {self.lastmove}')
            print(f'Reward: {self.p_reward} , {self.t_reward}')
            print(f'Objetivo: {self.objective}')
class YardRenderer:
    def __init__(self, yard_env):
        self.yard_env = yard_env
        self.num_stacks = self.yard_env.num_stacks
        self.stack_height = self.yard_env.stack_height
        self.square_size = 50
        self.margin = 10
        self.colors = {
            'empty': (255, 255, 255),
            'slab': (0, 0, 255),
            'robot': (255, 0, 0),
            'done': (255, 0, 255),
        }


    def render(self, mode='rgb_array'):
        yard_state = self.yard_env.state
        num_stacks = self.yard_env.num_stacks
        stack_height = self.yard_env.stack_height
        screen_width = num_stacks * (self.square_size + self.margin) + self.margin
        screen_height = (stack_height + 1) * (self.square_size + self.margin) + self.margin

        pygame.init()
        pygame.font.init()

        screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Yard")

        # Cria um objeto de fonte
        font = pygame.font.SysFont(None, 24)
        text_color = (0, 255, 0)
        text_obj_color  = (255, 0, 0)
        fim = None
        clock = pygame.time.Clock()

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                
                if mode == 'human':
                    # checking if keydown event happened or not
                    if event.type == pygame.KEYDOWN:
            
                        # Check for backspace
                        if event.key == pygame.K_m:
                            src_stack, dst_stack, num_slabs = input("Entre com o endereço de origem, destino, e quantidade de placas ").split()
                            action = int(src_stack), int(dst_stack), int(num_slabs)-1
                            observation, reward, done, info = self.yard_env.step(action)  # Execute a ação e receba o resultado

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
                stack = yard_state.stacks[stack_index]
                for slab_index in range(stack.get_size()):
                    slab = stack.stack[slab_index]
                    color = self.colors['slab']
                    x = self.margin + stack_index * (self.square_size + self.margin)
                    y = self.margin + (self.yard_env.stack_height - slab_index) * (self.square_size + self.margin)
                    rect = pygame.draw.rect(screen, color, (x, y, self.square_size, self.square_size))
                    if any(x == slab.slab_id for x in self.yard_env.objective) :
                        text = font.render(str(slab.slab_id), True, text_obj_color)
                    else:
                        text = font.render(str(slab.slab_id), True, text_color)
                    text_rect = text.get_rect(center=rect.center)
                    screen.blit(text, text_rect)

                for i in range(stack.get_size(), self.yard_env.stack_height):
                    color = self.colors['empty']
                    x = self.margin + stack_index * (self.square_size + self.margin)
                    y = self.margin + (self.yard_env.stack_height - i - 1) * (self.square_size + self.margin)
                    pygame.draw.rect(screen, color, (x, y, self.square_size, self.square_size))

            # for robot_index, robot_pos in enumerate(self.yard_env.robot_positions):
            #     x = self.margin + robot_pos * (self.square_size + self.margin)
            #     y = screen_height - self.margin - (robot_index + 1) * (self.square_size + self.margin)
            #     pygame.draw.rect(screen, self.colors['robot'], (x, y, self.square_size, self.square_size))
        
            if fim:
                color = self.colors['empty']
                text_obj_color = self.colors['empty']
                x = screen.get_width()/2
                y = screen.get_height()/2
                rect = pygame.draw.rect(screen, color, (x, y, 200, 200))
                text = font.render(str("FIM!!!! PARABENS"), True, text_color)
                text_rect = text.get_rect(center=rect.center)
                screen.blit(text, text_rect)

            pygame.display.flip()
            clock.tick(60)
        pygame.quit()