import random
import numpy as np
import pickle
from pomdp import *
from model import *
from model_phase2 import * 
import sys

with open(f'APT_data/hop.pickle','rb') as f:
    P0=pickle.load(f)
N_hop=[[] for i in range(7)]
for index_machine in list(P0):
    N_hop[int(P0[index_machine])].append(index_machine)
target=N_hop[0]
hop_1=N_hop[1]
hop_2=N_hop[2]
hop_3=N_hop[3]

def estimate_state(machine_state_list_belief_prability,cred_state_list_belief_prability):
    
    machine_state_list_estimated= [probability>0.79 for probability in machine_state_list_belief_prability]
    cred_state_list_estimated=[probability>0.81 for probability in cred_state_list_belief_prability]
    
    return machine_state_list_estimated,cred_state_list_estimated

def naive_estimate_state(naive_machine_state_list_estimated,observa_true):
    
    for i in range(len(naive_machine_state_list_estimated)):
        if i in observa_true:
            naive_machine_state_list_estimated[i]=True
    
    return naive_machine_state_list_estimated

def belief_state_update(my_pomdp_tem,machine_state_list_belief_prability,cred_state_list_belief_prability,action_contain_list,observation_machine,action_observation_list,observa_true):
    sampled_number=100
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

        elif machine_index_to_name(i) in hop_1+hop_2+hop_3: 
            if machine_state_list_belief_prability_new[i]<0.01:
                machine_state_list_belief_prability_new[i]=0.01
            elif machine_state_list_belief_prability_new[i]>0.8:
                machine_state_list_belief_prability_new[i]=0.8

        else: 
            if machine_state_list_belief_prability_new[i]<0.01:
                machine_state_list_belief_prability_new[i]=0.01
            elif machine_state_list_belief_prability_new[i]>0.05:
                machine_state_list_belief_prability_new[i]=0.05

    return machine_state_list_belief_prability_new, cred_state_list_belief_prability_new

if __name__ == "__main__":
    weight = float(sys.argv[1]) #10.0
    with open(f"./model_"+str(weight)+".pkl",'rb') as f:
        value_map_dict=pickle.load(f)

    with open(f"./model_phase2_"+str(weight)+".pkl",'rb') as f:
        value_map_dict_further=pickle.load(f)

    average_number=0
    times=0
    total_iteration=0
    total_containing_number=0
    result = {}
    estimate_high_error_lists=[]
    estimate_high_wrong=0
    estimate_high_right=0
    estimate_high_error=0

    naive_estimate_high_error_lists=[]
    naive_estimate_high_wrong=0
    naive_estimate_high_right=0
    naive_estimate_high_error=0

    for q in range(3):
        print("--------------------") 
        print(q)

        my_pomdp=POMDP()
        machine_state_list,cred_state_list,machine_state_list_belief_prability,cred_state_list_belief_prability=random_attacker_start(my_pomdp,seed=q)
        naive_machine_state_list_estimated=[False for i in range(len(machine_state_list))]
        observation_true_list=[]
        result[q] = [99999999999, -1]

        estimate_high_error_this=[]
        naive_estimate_high_error_this=[]

        for i in range(5000):
            machine_state_list_estimated,cred_state_list_estimated=estimate_state(machine_state_list_belief_prability,cred_state_list_belief_prability)
            higher_state_current_machine=full_state_to_higher_state(machine_state_list_estimated) 
            current_valuedic_key=higher_state_to_valuedic_key(higher_state_current_machine) 

            if current_valuedic_key in list(value_map_dict_further): 
                Q_value_current=value_map_dict_further[current_valuedic_key] 
            else:
                simplest_state_current_machine=full_state_to_simplest_state(machine_state_list_estimated)
                current_valuedic_key=simplest_state_to_valuedic_key(simplest_state_current_machine)
                Q_value_current=value_map_dict[current_valuedic_key]

            estimate_high=np.array(list(map(int, full_state_to_higher_state(machine_state_list_estimated))))
            true_high=np.array(list(map(int,full_state_to_higher_state(machine_state_list))))
            print(np.sum(np.abs(estimate_high-true_high)))
            #estimate_low=np.array(list(map(int,full_state_to_simplest_state(machine_state_list_estimated))))
            #estimate_lowlist=np.array(list(map(int,full_state_to_simplest_state(machine_state_list))))
            #print(np.sum(np.abs(estimate_low-estimate_lowlist)))
            if np.sum(np.abs(estimate_high-true_high))>0:
                estimate_high_wrong=estimate_high_wrong+1
                estimate_high_error+=np.sum(np.abs(estimate_high-true_high))
            else:
                estimate_high_right=estimate_high_right+1
            estimate_high_error_this.append(np.sum(np.abs(estimate_high-true_high)))
 
            naive_estimate_high=np.array(list(map(int, full_state_to_higher_state(naive_machine_state_list_estimated))))
            print(np.sum(np.abs(naive_estimate_high-true_high)))
            if np.sum(np.abs(naive_estimate_high-true_high))>0:
                naive_estimate_high_wrong=naive_estimate_high_wrong+1
                naive_estimate_high_error+=np.sum(np.abs(naive_estimate_high-true_high))
            else:
                naive_estimate_high_right=naive_estimate_high_right+1
            naive_estimate_high_error_this.append(np.sum(np.abs(naive_estimate_high-true_high)))


            action_index=Q_value_current.index(max(Q_value_current)) 
            #action_index=0
            action_contain_list=index_to_action(action_index)

            #state_transition
            machine_state_list_new,cred_state_list_new=my_pomdp.state_transition(machine_state_list,cred_state_list,action_contain_list)
            
            machine_state_list=machine_state_list_new
            cred_state_list=cred_state_list_new

            observation_list=[machine_name_to_index(ele) for ele in hop_1+hop_2+hop_3]
            action_observation_list=random.sample(observation_list,2)

            observation_machine=my_pomdp.state_observation(machine_state_list,action_observation_list) 
            print(action_observation_list)
            print(observation_machine)

            for qq in range(len(observation_machine)):
                if observation_machine[qq]==True:
                    observation_true_list.append(action_observation_list[qq])

            print(machine_state_list_belief_prability[action_observation_list[0]],machine_state_list_belief_prability[action_observation_list[1]])

            my_pomdp_tem=POMDP()
            machine_state_list_belief_prability,cred_state_list_belief_prability=belief_state_update(my_pomdp_tem,machine_state_list_belief_prability,cred_state_list_belief_prability,action_contain_list,observation_machine,action_observation_list,observation_true_list)
            naive_machine_state_list_estimated=naive_estimate_state(naive_machine_state_list_estimated,observation_true_list)
            
            #print(machine_state_list_belief_prability)
            #print([machine_state_list_belief_prability[i] for i in range(len(machine_state_list_belief_prability)) if machine_index_to_name(i) in hop_1+hop_2+hop_3])
            print(machine_state_list_belief_prability[action_observation_list[0]],machine_state_list_belief_prability[action_observation_list[1]])
            print(i) 
            total_iteration=total_iteration+1
            total_containing_number+=len(action_contain_list)
            print("------------reminder_---------------")
            print(str(times)+"/"+str(q))
            if times>0:
                print(1.0*average_number/times)
            print("------------next____________")

            machine_has_compr=[index for index in range(len(machine_state_list_new)) if machine_state_list_new[index]==True] 
            machine_has_compr_hop=[my_pomdp.hop[machine_index_to_name(index)] for index in machine_has_compr]
            result[q][0] = min(result[q][0], min(machine_has_compr_hop)) 
            if 0 in machine_has_compr_hop:
                continue
        result[q][1] = i
        estimate_high_error_lists.append(estimate_high_error_this)

    print(estimate_high_error_lists)
    print(estimate_high_wrong)
    print(estimate_high_right)
    print(estimate_high_error)

    print(naive_estimate_high_error_lists)
    print(naive_estimate_high_wrong)
    print(naive_estimate_high_right)
    print(naive_estimate_high_error)