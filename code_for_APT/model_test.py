import random
import numpy as np
import pickle
from pomdp import *
from model import *
from model_phase2 import * 




if __name__ == "__main__":
    with open(f'./model.pkl','rb') as f:
        value_map_dict=pickle.load(f)

    with open(f'./model_phase2.pkl','rb') as f:
        value_map_dict_further=pickle.load(f)


    average_number=0
    for q in range(200):
        print("--------------------") 
        print(q)

        my_pomdp=POMDP()
        machine_state_list,cred_state_list,machine_state_list_belief_prability,cred_state_list_belief_prability=random_attacker_start(my_pomdp,seed=q%20)
        
        for i in range(10000):
            #choose action based on eps-greedy policy
            higher_state_current_machine,higher_state_current_cred=full_state_to_higher_state(machine_state_list,cred_state_list)
            current_valuedic_key=higher_state_to_valuedic_key(higher_state_current_machine,higher_state_current_cred)

            if current_valuedic_key in list(value_map_dict_further):
                Q_value_new=value_map_dict_further[current_valuedic_key] 
            else:
                simplest_state_current_machine,simplest_state_current_cred=full_state_to_simplest_state(machine_state_list,cred_state_list)
                current_valuedic_key=simplest_state_to_valuedic_key(simplest_state_current_machine,simplest_state_current_cred)
                Q_value_current=value_map_dict[current_valuedic_key]

            action_index=Q_value_current.index(max(Q_value_current))
            action_contain_list=index_to_action(action_index)

            #state_transition
            machine_state_list_new,cred_state_list_new=my_pomdp.state_transition(machine_state_list,cred_state_list,action_contain_list)
            
            machine_state_list=machine_state_list_new
            cred_state_list=cred_state_list_new
            print(i) 

            if 0 in machine_has_compr_hop:
                print(value_map_dict[current_valuedic_key])
                average_number+=i
                break

    average_number=average_number/200.0
    print(average_number)
