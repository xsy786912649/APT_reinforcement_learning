import random
import numpy as np
import pickle
from pomdp import *
from model import *
from model_phase2 import * 

with open(f'./APT_data/hop.pickle','rb') as f:
    P0=pickle.load(f)
N_hop=[[] for i in range(7)]
for index_machine in list(P0):
    N_hop[int(P0[index_machine])].append(index_machine)
target=N_hop[0]
hop_1=N_hop[1]
hop_2=N_hop[2]


if __name__ == "__main__":
    with open(f'./model.pkl','rb') as f:
        value_map_dict=pickle.load(f)

    with open(f'./model_phase2.pkl','rb') as f:
        value_map_dict_further=pickle.load(f)

    average_number=0
    times=0
    for q in range(100):
        print("--------------------") 
        print(q)

        my_pomdp=POMDP()
        machine_state_list,cred_state_list,machine_state_list_belief_prability,cred_state_list_belief_prability=random_attacker_start(my_pomdp,seed=q)
        
        for i in range(5000):
            
            action_index=random.randint(0, 1+14+7*13-1)
            action_contain_list=index_to_action(action_index)

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
                break

    average_number=average_number/times
    print(average_number)
    print(times)
