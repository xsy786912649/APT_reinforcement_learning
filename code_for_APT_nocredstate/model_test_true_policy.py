import random
import numpy as np
import pickle
from pomdp import *
from model import *
from model_phase2 import * 
from utils import parse

with open(f'./APT_data/hop.pickle','rb') as f:
    P0=pickle.load(f)
N_hop=[[] for i in range(7)]
for index_machine in list(P0):
    N_hop[int(P0[index_machine])].append(index_machine)
target=N_hop[0]
hop_1=N_hop[1]
hop_2=N_hop[2]



def test():
    with open(f'./model.pkl','rb') as f:
        value_map_dict=pickle.load(f)

    with open(f'./model_phase2.pkl','rb') as f:
        value_map_dict_further=pickle.load(f)

    average_number=0
    times=0
    available = [1, 9, 11, 15, 16, 17, 19, 22, 32, 34, 38, 40, 44, 45, 50, 64, 66, 67, 71, 73, 75, 76, 78, 79, 80, 86, 88, 90, 91, 93, 95, 96, 98, 100, 103, 105, 108, 109, 112, 116, 119, 122, 123, 128, 129, 130, 131, 136, 138, 140, 143, 145, 146, 150, 151, 153, 154, 156, 161, 163, 169, 170, 171, 172, 183, 184, 186, 188, 189, 193, 195, 198, 200, 202, 203, 205, 208, 210, 212, 220, 225, 226, 230, 232, 234, 239, 241, 244, 245, 247, 250, 253, 254, 256, 258, 260, 261, 262, 266, 269, 272, 274, 275, 276, 278, 279, 281, 284, 285, 286, 289, 292, 295, 297, 298, 299, 300, 306, 307, 309, 311, 315, 321, 330, 332, 335, 338, 341, 342, 346, 347, 348, 349, 350, 354, 356, 357, 359, 361, 363, 366, 367, 368, 375, 376, 379, 382, 385, 386, 387, 390, 392, 395]

    success = {}
    for i in available:
        success[i] = []
    for q in available:
        print("--------------------") 
        print(q)

        for _ in range(10):
            check = False
            my_pomdp=POMDP()
            machine_state_list,cred_state_list,machine_state_list_belief_prability,cred_state_list_belief_prability=random_attacker_start(my_pomdp,seed=q)
        
            for i in range(5000):
                
                action_contain_list1=[i for i in range(len(machine_state_list)) if machine_state_list[i]==True and machine_index_to_name(i) in hop_1] 
                if len(action_contain_list1)<3:
                    action_contain_list=action_contain_list1
                else:
                    action_contain_list=[action_contain_list1[0],action_contain_list1[1]]

                #state_transition
                machine_state_list_new,cred_state_list_new=my_pomdp.state_transition(machine_state_list,cred_state_list,action_contain_list)
                
                machine_state_list=machine_state_list_new
                cred_state_list=cred_state_list_new

                machine_has_compr=[index for index in range(len(machine_state_list_new)) if machine_state_list_new[index]==True] 
                machine_has_compr_hop=[my_pomdp.hop[machine_index_to_name(index)] for index in machine_has_compr] 
                if 0 in machine_has_compr_hop:
                    average_number+=i
                    times+=1
                    print(i)
                    success[q].append(i)
                    check = True
                    break
            if not check:
                success[q].append(5000)
    average_number=average_number/times
    print(average_number)
    print(times)
    print(success)

if __name__ == "__main__":
    args = parse()
    if args.test:
        test()
        exit()
        
    with open(f'./model.pkl','rb') as f:
        value_map_dict=pickle.load(f)

    with open(f'./model_phase2.pkl','rb') as f:
        value_map_dict_further=pickle.load(f)

    average_number=0
    times=0
    available = []
    for q in range(400):
        print("--------------------") 
        print(q)

        my_pomdp=POMDP()
        machine_state_list,cred_state_list,machine_state_list_belief_prability,cred_state_list_belief_prability=random_attacker_start(my_pomdp,seed=q)
        
        for i in range(5000):
            

            action_contain_list1=[i for i in range(len(machine_state_list)) if machine_state_list[i]==True and machine_index_to_name(i) in hop_1] 
            if len(action_contain_list1)<3:
                action_contain_list=action_contain_list1
            else:
                action_contain_list=[action_contain_list1[0],action_contain_list1[1]]

            #state_transition
            machine_state_list_new,cred_state_list_new=my_pomdp.state_transition(machine_state_list,cred_state_list,action_contain_list)
            
            machine_state_list=machine_state_list_new
            cred_state_list=cred_state_list_new

            machine_has_compr=[index for index in range(len(machine_state_list_new)) if machine_state_list_new[index]==True] 
            machine_has_compr_hop=[my_pomdp.hop[machine_index_to_name(index)] for index in machine_has_compr] 
            if 0 in machine_has_compr_hop:
                average_number+=i
                times+=1
                print(i)
                if i == 0:
                    break
                available.append(q)
                break

    average_number=average_number/times
    print(average_number)
    print(times)
    print(available)
