import random
import numpy as np
import pickle
from pomdp import *
from model import *
import os
import copy
import sys

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

def higher_state_to_valuedic_key(higher_state_current_machine):
    return tuple(higher_state_current_machine)

if __name__ == "__main__":
    weight = float(sys.argv[1]) #10.0 
    base_penalty= float(sys.argv[2])
    with open(f"./model_"+str(weight)+".pkl",'rb') as f:
        value_map_dict=pickle.load(f)

    if os.path.exists("./model_phase2_"+str(weight)+".pkl"):
        with open(f"./model_phase2_"+str(weight)+".pkl",'rb') as f:
            value_map_dict_further=pickle.load(f)
    else:
        value_map_dict_further={}

    for q in range(100000):
        print("--------------------") 
        print(q)

        my_pomdp=POMDP()
        machine_state_list,cred_state_list,machine_state_list_belief_prability,cred_state_list_belief_prability=random_attacker_start(my_pomdp)
        
        no_change_action=False
        action_index=0
        action_contain_list=[]

        for i in range(2000):
            #choose action based on eps-greedy policy
            higher_state_current_machine=full_state_to_higher_state(machine_state_list)
            current_valuedic_key=higher_state_to_valuedic_key(higher_state_current_machine)
            
            if current_valuedic_key in list(value_map_dict_further):
                Q_value_current=value_map_dict_further[current_valuedic_key]   
            else:
                simplest_state_current_machine=full_state_to_simplest_state(machine_state_list)
                simplest_current_valuedic_key=simplest_state_to_valuedic_key(simplest_state_current_machine)
                value_map_dict_further[current_valuedic_key]=copy.deepcopy(value_map_dict[simplest_current_valuedic_key])
                Q_value_current=value_map_dict_further[current_valuedic_key]

            if no_change_action==False:
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
                higher_state_new_machine=full_state_to_higher_state(machine_state_list_new)
                new_valuedic_key_higher=higher_state_to_valuedic_key(higher_state_new_machine)
                if new_valuedic_key_higher in list(value_map_dict_further):
                    Q_value_new=value_map_dict_further[new_valuedic_key_higher] 
                else:
                    simplest_state_new_machine=full_state_to_simplest_state(machine_state_list_new)
                    new_valuedic_key_simplest=simplest_state_to_valuedic_key(simplest_state_new_machine)
                    Q_value_new=value_map_dict[new_valuedic_key_simplest] 
                    
                reward_safe=0.0
                if 0 in machine_has_compr_hop_new:
                    reward_safe=-200.0
                reward_avai=False
                if len(action_contain_list)==1:
                    if machine_index_to_name(action_contain_list[0]) in hop_1:
                        reward_avai=-weight
                    elif machine_index_to_name(action_contain_list[0]) in hop_2:
                        reward_avai=-base_penalty
                elif len(action_contain_list)==2:
                    if (machine_index_to_name(action_contain_list[0]) in hop_1) and (machine_index_to_name(action_contain_list[1]) in hop_1):
                        reward_avai=-2*weight
                    elif (machine_index_to_name(action_contain_list[0]) in hop_2) and (machine_index_to_name(action_contain_list[1]) in hop_2):
                        reward_avai=-2*base_penalty
                    else:
                        reward_avai=-base_penalty-weight
                reward=reward_safe+reward_avai
                if 0 not in machine_has_compr_hop_new:
                    value_map_dict_further[current_valuedic_key][action_index]=value_map_dict_further[current_valuedic_key][action_index]*(1-lr)+lr*(reward+499.0/500*max(Q_value_new))
                else:
                    value_map_dict_further[current_valuedic_key][action_index]=value_map_dict_further[current_valuedic_key][action_index]*(1-lr)+lr*(reward-2000.0)

                print(higher_state_current_machine)
            
            if full_state_to_higher_state(machine_state_list)==full_state_to_higher_state(machine_state_list_new):
                no_change_action=True
            else:
                no_change_action=False

            machine_state_list=machine_state_list_new
            cred_state_list=cred_state_list_new
            print(i) 

            if 0 in machine_has_compr_hop_new:
                break

            #print(full_state_to_simplest_state(machine_state_list,cred_state_list))
            #print(full_state_to_higher_state(machine_state_list,cred_state_list))

        if q%100==0:
            f_save=open("model_phase2_"+str(weight)+".pkl",'wb')
            pickle.dump(value_map_dict_further,f_save)
            f_save.close()