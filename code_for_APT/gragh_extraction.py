import networkx
import math
import numpy as np
import pickle
import random

G=0
with open(f'./APT_data/network_topo.gpickle','rb') as f:
    G=pickle.load(f)
node_dic=G.nodes
machine_list=list(node_dic)
machine_number=len(machine_list)

obtained_cred=[]
using_cred_stored=[]

stored_cred=[]
with open(f'./APT_data/comp_cred.pickle','rb') as f:
    P0=pickle.load(f)
for index_machine in list(P0):
    stored_cred.append(P0[index_machine] )

with open(f'./APT_data/highest_level_credential.pickle','rb') as f:
    P0=pickle.load(f)
highest_level_credential_list=list(P0)
with open(f'./APT_data/lowest_level_credential.pickle','rb') as f:
    P0=pickle.load(f)
lowest_level_credential_list=list(P0)
with open(f'./APT_data/middle_level_credential.pickle','rb') as f:
    P0=pickle.load(f)
middle_level_credential_list=list(P0)

credential_list=highest_level_credential_list+middle_level_credential_list+lowest_level_credential_list
credential_list.sort(key=lambda x: int(x[4:]))
credential_number=len(credential_list)
print(machine_number)
print(credential_number)

hop=0
with open(f'./APT_data/hop.pickle','rb') as f:
    hop=pickle.load(f)
N_hop=[[] for i in range(7)]
for index_machine in list(hop):
    N_hop[int(hop[index_machine])].append(index_machine)

def machine_index_to_name(index):
    return 'comp'+str(index)

def machine_name_to_index(name):
    return int(name[4:])

def cred_name_to_index(name):
    return int(name[4:])

def cred_index_to_name(name):
    return 'cred'+str(name)

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
 
    return False

def state_observation(machine_state_list=[False]*machine_number,cred_state_list=[False]*credential_number,action_observation_list=[]):
    global obtained_cred
    global using_cred_stored

    if not action_observation_list == []:
        assert is_number(action_observation_list[0])
    observation_machine=[machine_state_list[machine_index] for machine_index in action_observation_list]

    obtained_cred_obervation=[]
    using_cred_obervation=[]
    non_obtained_cred_observation=[]
    for machine_index in action_observation_list:
        obtained_cred_obervation.append(set(obtained_cred[machine_index]))
        using_cred_obervation.append(set(using_cred_stored[machine_index]))
        non_obtained_cred_observation.append([cred for cred in stored_cred[machine_index] if cred not in obtained_cred[machine_index]])

        #print(observation_machine)
        #print(obtained_cred_obervation)
        #print(using_cred_obervation)
        #print(non_obtained_cred_observation)
        #print(stored_cred[machine_index])

    return observation_machine,obtained_cred_obervation,using_cred_obervation,non_obtained_cred_observation

def state_transition(machine_state_list=[False]*machine_number,cred_state_list=[False]*credential_number,action_contain_list=[]):
    global obtained_cred
    global using_cred_stored

    if not action_contain_list == []:
        assert is_number(action_contain_list[0])

    contain_machine_name_list=[machine_index_to_name(index) for index in action_contain_list]
    available_cred= [cred_index_to_name(index) for index in range(len(cred_state_list)) if cred_state_list[index]==True]

    for n in range(len(machine_state_list)):
        if machine_state_list[n]==True and (n not in action_contain_list):
            neighbors_of_n_list=list(G.neighbors(machine_list[n]))
            neighbors_of_n_list_noncompromised=[machine for machine in neighbors_of_n_list if machine_state_list[machine_name_to_index(machine)]==False]
            potential_plan_compromise_list = [item for item in neighbors_of_n_list_noncompromised]
            if potential_plan_compromise_list==[]:
                continue

            plan_compromise_machine=random.choice(potential_plan_compromise_list)

            if plan_compromise_machine in contain_machine_name_list:
                continue

            using_cred=random.choice(available_cred)
            good_cred_list=list(node_dic[plan_compromise_machine])
            if using_cred not in good_cred_list:
                continue
            elif using_cred in good_cred_list:
                machine_state_list[machine_name_to_index(plan_compromise_machine)]=True
                may_obtain_cred= node_dic[plan_compromise_machine][using_cred]
                may_obtain_cred_index=[cred_name_to_index(cred_name) for cred_name in may_obtain_cred]

                obtained_cred_this_machine=[]
                if using_cred in highest_level_credential_list:
                    for index in may_obtain_cred_index:
                        cred_state_list[index]=True
                        obtained_cred_this_machine.append(cred_index_to_name(index))
                elif using_cred in middle_level_credential_list:
                    for index in may_obtain_cred_index:
                        if np.random.rand(1)<0.7:
                            cred_state_list[index]=True
                            obtained_cred_this_machine.append(cred_index_to_name(index))
                elif using_cred in lowest_level_credential_list:
                    for index in may_obtain_cred_index:
                        if np.random.rand(1)<0.4:
                            cred_state_list[index]=True
                            obtained_cred_this_machine.append(cred_index_to_name(index))
                            
                using_cred_stored[machine_name_to_index(plan_compromise_machine)].append(using_cred)
                obtained_cred[machine_name_to_index(plan_compromise_machine)].extend(obtained_cred_this_machine.copy())

    return machine_state_list,cred_state_list

def reset():
    global obtained_cred
    global using_cred_stored
    obtained_cred=[[] for i in range(machine_number)]
    using_cred_stored=[[] for i in range(machine_number)]

    return

def set_initial(obtained_cred_inital,using_cred_stored_initial):
    global obtained_cred
    global using_cred_stored
    obtained_cred=obtained_cred_inital
    using_cred_stored=using_cred_stored_initial

    return

if __name__ == "__main__":
    initial_compro_machine_index=11
    initial_using_cred_index=1

    reset()

    machine_state_list=[False]*machine_number
    machine_state_list[initial_compro_machine_index]=True

    obtained_cred_inital=[[] for i in range(machine_number)]
    using_cred_stored_initial=[[] for i in range(machine_number)]
    original_cred_used=list(node_dic[machine_index_to_name(initial_compro_machine_index)])[initial_using_cred_index]
    using_cred_stored_initial[initial_compro_machine_index]=[original_cred_used]
    original_cred_obtained=node_dic[machine_index_to_name(initial_compro_machine_index)][original_cred_used]
    obtained_cred_inital[initial_compro_machine_index]=original_cred_obtained
    set_initial(obtained_cred_inital,using_cred_stored_initial)

    cred_state_list_index=[cred_name_to_index(cred) for cred in original_cred_obtained]
    cred_state_list=[False]*credential_number

    for index in cred_state_list_index:
        cred_state_list[index]=True
    action_contain_list=[]
    #action_contain_list=[145,361,499,199,246,398,252,102,296,442,155,347,427,320,287]

    for i in range(5000):
        machine_state_list,cred_state_list=state_transition(machine_state_list,cred_state_list,action_contain_list)
        machine_has_compr=[index for index in range(len(machine_state_list)) if machine_state_list[index]==True]
        machine_has_compr_hop=[hop[machine_index_to_name(index)] for index in machine_has_compr]
        print("--------------------")
        print(i)
        print(machine_has_compr)
        print(machine_has_compr_hop)
        
        if 0 in machine_has_compr_hop:
            #print([machine_has_compr[j] for j in range(len(machine_has_compr)) if machine_has_compr_hop[j]==1])
            input()

        action_observation_list=[0]
        state_observation(machine_state_list,cred_state_list,action_observation_list) 
        
        #print(obtained_cred)
        #if len([index for index in range(len(machine_state_list)) if machine_state_list[index]==True])>1:
        #    input()
        #print([index for index in range(len(cred_state_list)) if cred_state_list[index]==True]) 

    