import random
import numpy as np
import pickle
from pomdp import *
from model import *
import os
import copy

with open(f'./APT_data/hop.pickle','rb') as f:
    P0=pickle.load(f)
N_hop=[[] for i in range(7)]
for index_machine in list(P0):
    N_hop[int(P0[index_machine])].append(index_machine)
target=N_hop[0]
hop_1=N_hop[1]
hop_2=N_hop[2]
my_pomdp1=POMDP()

eps=0.3
lr=0.5

def higher_state_to_valuedic_key(higher_state_current_machine,higher_state_current_cred):
    return tuple(higher_state_current_machine+higher_state_current_cred)

if __name__ == "__main__":
    with open(f'./model.pkl','rb') as f:
        value_map_dict=pickle.load(f)

    if os.path.exists('./model_phase2.pkl'):
        with open(f'./model_phase2.pkl','rb') as f:
            value_map_dict_further=pickle.load(f)
    else:
        value_map_dict_further={}

    for q in range(10000):
        print("--------------------") 
        print(q)

        my_pomdp=POMDP()
        machine_state_list,cred_state_list,machine_state_list_belief_prability,cred_state_list_belief_prability=random_attacker_start(my_pomdp)
        
        for i in range(5000):
            #choose action based on eps-greedy policy
            higher_state_current_machine,higher_state_current_cred=full_state_to_higher_state(machine_state_list,cred_state_list)
            current_valuedic_key=higher_state_to_valuedic_key(higher_state_current_machine,higher_state_current_cred)
            
            if current_valuedic_key in list(value_map_dict_further):
                Q_value_current=value_map_dict_further[current_valuedic_key]   
            else:
                simplest_state_current_machine,simplest_state_current_cred=full_state_to_simplest_state(machine_state_list,cred_state_list)
                simplest_current_valuedic_key=simplest_state_to_valuedic_key(simplest_state_current_machine,simplest_state_current_cred)
                value_map_dict_further[current_valuedic_key]=copy.deepcopy(value_map_dict[simplest_current_valuedic_key])
                Q_value_current=value_map_dict_further[current_valuedic_key]

            action_index=0
            if np.random.rand(1)<eps:
                action_index=random.randint(0, len(Q_value_current)-1)
            else:
                action_index=Q_value_current.index(max(Q_value_current))
            action_contain_list=index_to_action(action_index)

            #state_transition
            machine_state_list_new,cred_state_list_new=my_pomdp.state_transition(machine_state_list,cred_state_list,action_contain_list)
            
            #if the state is interesting, update the q table
            machine_has_compr_new=[index for index in range(len(machine_state_list_new)) if machine_state_list_new[index]==True] 
            machine_has_compr_hop_new=[my_pomdp.hop[machine_index_to_name(index)] for index in machine_has_compr_new] 
            if (0 in machine_has_compr_hop_new) or (1 in machine_has_compr_hop_new) or (2 in machine_has_compr_hop_new): 
                higher_state_new_machine,higher_state_new_cred=full_state_to_higher_state(machine_state_list_new,cred_state_list_new)
                new_valuedic_key_higher=higher_state_to_valuedic_key(higher_state_new_machine,higher_state_new_cred)
                if new_valuedic_key_higher in list(value_map_dict_further):
                    Q_value_new=value_map_dict_further[new_valuedic_key_higher] 
                else:
                    simplest_state_new_machine,simplest_state_new_cred=full_state_to_simplest_state(machine_state_list_new,cred_state_list_new)
                    new_valuedic_key_simplest=simplest_state_to_valuedic_key(simplest_state_new_machine,simplest_state_new_cred)
                    Q_value_new=value_map_dict[new_valuedic_key_simplest] 
                    
                reward_safe=0.0
                if 0 in machine_has_compr_hop_new:
                    reward_safe=-100.0
                reward_avai=float(len(action_contain_list))*(-0.1)
                reward=reward_safe+reward_avai
                if 0 not in machine_has_compr_hop_new:
                    value_map_dict_further[current_valuedic_key][action_index]=value_map_dict_further[current_valuedic_key][action_index]*(1-lr)+lr*(reward+4999.0/5000*max(Q_value_new))
                else:
                    value_map_dict_further[current_valuedic_key][action_index]=value_map_dict_further[current_valuedic_key][action_index]*(1-lr)+lr*(reward-4990.0)

                print(higher_state_current_machine,higher_state_current_cred)
            
            machine_state_list=machine_state_list_new
            cred_state_list=cred_state_list_new
            print(i) 

            if 0 in machine_has_compr_hop_new:
                break

            #print(full_state_to_simplest_state(machine_state_list,cred_state_list))
            #print(full_state_to_higher_state(machine_state_list,cred_state_list))

        if q%100==0:
            f_save=open("model_phase2.pkl",'wb')
            pickle.dump(value_map_dict_further,f_save)
            f_save.close()