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
hop_3=N_hop[3]

def estimate_state(machine_state_list_belief_prability,cred_state_list_belief_prability):
    
    machine_state_list_estimated= [probability>0.6 for probability in machine_state_list_belief_prability]
    cred_state_list_estimated=[probability>0.6 for probability in cred_state_list_belief_prability]
    
    return machine_state_list_estimated,cred_state_list_estimated

def belief_state_update(my_pomdp_tem,machine_state_list_belief_prability,cred_state_list_belief_prability,action_contain_list,observation_machine,obtained_cred_obervation,action_observation_list,observa_true):
    sampled_number=20
    machine_state_list_belief_prability_new=np.zeros_like(machine_state_list_belief_prability)
    cred_state_list_belief_prability_new=np.zeros_like(cred_state_list_belief_prability)
    i=0
    while i<sampled_number:
        machine_state_list_sampled= [np.random.rand(1)<probablity for probablity in machine_state_list_belief_prability]
        cred_state_list_sampled = [np.random.rand(1)<probablity for probablity in cred_state_list_belief_prability]
        machine_state_list_new_sampled,cred_state_list_new_sampled=my_pomdp_tem.state_transition_temp(machine_state_list_sampled,cred_state_list_sampled,action_contain_list)
        if machine_state_list_new_sampled[action_observation_list[0]]==observation_machine[0] and machine_state_list_new_sampled[action_observation_list[1]]==observation_machine[1]:
            machine_state_list_belief_prability_new+=np.array([1.0/sampled_number*int(ele) for ele in machine_state_list_new_sampled])
            cred_state_list_belief_prability_new+=np.array([1.0/sampled_number*int(ele) for ele in cred_state_list_new_sampled])
            i=i+1

    for i in range(len(machine_state_list_belief_prability_new)):
        if i in observa_true:
            machine_state_list_belief_prability_new[i]=1.0

        elif machine_index_to_name(i) in N_hop[1]+N_hop[2] : 
            if machine_state_list_belief_prability_new[i]<0.03:
                machine_state_list_belief_prability_new[i]=0.03
            elif machine_state_list_belief_prability_new[i]>0.8:
                machine_state_list_belief_prability_new[i]=0.8

        else: 
            if machine_state_list_belief_prability_new[i]<0.03:
                machine_state_list_belief_prability_new[i]=0.03
            elif machine_state_list_belief_prability_new[i]>0.3:
                machine_state_list_belief_prability_new[i]=0.3

    for ele in obtained_cred_obervation[0]+obtained_cred_obervation[1]:
        cred_state_list_belief_prability_new[cred_name_to_index(ele)]=1.0

    return machine_state_list_belief_prability_new, cred_state_list_belief_prability_new

if __name__ == "__main__":
    with open(f'./model.pkl','rb') as f:
        value_map_dict=pickle.load(f)

    with open(f'./model_phase2.pkl','rb') as f:
        value_map_dict_further=pickle.load(f)

    average_number=0
    times=0
    total_iteration=0
    total_containing_number=0
    for q in range(100):
        print("--------------------") 
        print(q)

        my_pomdp=POMDP()
        machine_state_list,cred_state_list,machine_state_list_belief_prability,cred_state_list_belief_prability=random_attacker_start(my_pomdp,seed=q)
        observation_true_list=[]

        for i in range(5000):
            machine_state_list_estimated=machine_state_list
            cred_state_list_estimated=cred_state_list
            higher_state_current_machine,higher_state_current_cred=full_state_to_higher_state(machine_state_list_estimated,cred_state_list_estimated) 
            current_valuedic_key=higher_state_to_valuedic_key(higher_state_current_machine,higher_state_current_cred) 

            if current_valuedic_key in list(value_map_dict_further): 
                Q_value_current=value_map_dict_further[current_valuedic_key] 
            else:
                simplest_state_current_machine,simplest_state_current_cred=full_state_to_simplest_state(machine_state_list_estimated,cred_state_list_estimated)
                current_valuedic_key=simplest_state_to_valuedic_key(simplest_state_current_machine,simplest_state_current_cred)
                Q_value_current=value_map_dict[current_valuedic_key]

            action_index=Q_value_current.index(max(Q_value_current)) 
            #action_index=0
            action_contain_list=index_to_action(action_index)

            #state_transition
            machine_state_list_new,cred_state_list_new=my_pomdp.state_transition(machine_state_list,cred_state_list,action_contain_list)
            
            machine_state_list=machine_state_list_new
            cred_state_list=cred_state_list_new

            #print(i) 
            total_iteration=total_iteration+1
            total_containing_number+=len(action_contain_list)
            #print("------------reminder_---------------")
            #print(str(times)+"/"+str(q))
            #if times>0:
            #    print(1.0*average_number/times)
            #print("------------next____________")

            machine_has_compr=[index for index in range(len(machine_state_list_new)) if machine_state_list_new[index]==True] 
            machine_has_compr_hop=[my_pomdp.hop[machine_index_to_name(index)] for index in machine_has_compr] 
            if 0 in machine_has_compr_hop:
                average_number+=i
                times+=1
                print(i)
                break

    average_number=average_number/times
    total_containing_number_fre=total_containing_number/total_iteration
    print(average_number)
    print(total_containing_number_fre)
    print(times)
