import networkx
import math
import numpy as np
import pickle
import random
import copy

with open(f'./APT_data/hop.pickle','rb') as f:
    P0=pickle.load(f)
N_hop=[[] for i in range(7)]
for index_machine in list(P0):
    N_hop[int(P0[index_machine])].append(index_machine)
All_machine=N_hop[0]+N_hop[1]+N_hop[2]+N_hop[3]+N_hop[4]+N_hop[5]+N_hop[6]

def machine_index_to_name(index):
    return All_machine[index]

def machine_name_to_index(name):
    return All_machine.index(name)

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

class POMDP:
    def __init__(self):
        self.obtained_cred=[]
        self.using_cred_stored=[]

        self.G=0
        with open(f'./APT_data/network_topo.gpickle','rb') as f:
            self.G=pickle.load(f)
        self.node_dic=self.G.nodes
        self.machine_list=list(self.node_dic)
        self.machine_number=len(self.machine_list)

        self.stored_cred=[]
        with open(f'./APT_data/comp_cred.pickle','rb') as f:
            P0=pickle.load(f)
        for index_machine in list(P0):
            self.stored_cred.append(P0[index_machine] )

        with open(f'./APT_data/highest_level_credential.pickle','rb') as f:
            P0=pickle.load(f)
        self.highest_level_credential_list=list(P0)
        with open(f'./APT_data/lowest_level_credential.pickle','rb') as f:
            P0=pickle.load(f)
        self.lowest_level_credential_list=list(P0)
        with open(f'./APT_data/middle_level_credential.pickle','rb') as f:
            P0=pickle.load(f)
        self.middle_level_credential_list=list(P0)

        credential_list=self.highest_level_credential_list+self.middle_level_credential_list+self.lowest_level_credential_list
        credential_list.sort(key=lambda x: int(x[4:]))
        self.credential_number=len(credential_list)
        #print(self.machine_number)
        #print(self.credential_number)

        self.hop=0
        with open(f'./APT_data/hop.pickle','rb') as f:
            self.hop=pickle.load(f)
        self.N_hop=[[] for i in range(7)]
        for index_machine in list(self.hop):
            self.N_hop[int(self.hop[index_machine])].append(index_machine)

    def state_observation(self,machine_state_list,action_observation_list=[]):

        if not action_observation_list == []:
            assert is_number(action_observation_list[0])
        observation_machine=[machine_state_list[machine_index] for machine_index in action_observation_list]

        #obtained_cred_obervation=[]
        #using_cred_obervation=[]
        #non_obtained_cred_observation=[]
        #for machine_index in action_observation_list:
        #    obtained_cred_obervation.append(list(set(self.obtained_cred[machine_index])))
        #    using_cred_obervation.append(list(set(self.using_cred_stored[machine_index])))
        #    non_obtained_cred_observation.append([cred for cred in self.stored_cred[machine_index] if cred not in self.obtained_cred[machine_index]])

        return observation_machine

    def state_transition(self,machine_state_list,cred_state_list,action_contain_list=[]):
        
        machine_state_list=copy.deepcopy(machine_state_list)
        cred_state_list=copy.deepcopy(cred_state_list)

        if (not (True in machine_state_list)) or (not (True in cred_state_list)):
            return machine_state_list,cred_state_list

        if not action_contain_list == []:
            assert is_number(action_contain_list[0])

        contain_machine_name_list=[machine_index_to_name(index) for index in action_contain_list]
        available_cred= [cred_index_to_name(index) for index in range(len(cred_state_list)) if cred_state_list[index]==True]

        for n in range(len(machine_state_list)):
            if machine_state_list[n]==True and (n not in action_contain_list):
                neighbors_of_n_list=list(self.G.neighbors(self.machine_list[n]))
                neighbors_of_n_list_noncompromised=[machine for machine in neighbors_of_n_list if machine_state_list[machine_name_to_index(machine)]==False]
                potential_plan_compromise_list = [item for item in neighbors_of_n_list_noncompromised]
                if potential_plan_compromise_list==[]:
                    continue

                plan_compromise_machine=random.choice(potential_plan_compromise_list)

                if plan_compromise_machine in contain_machine_name_list:
                    continue

                using_cred=random.choice(available_cred)
                good_cred_list=list(self.node_dic[plan_compromise_machine])
                if using_cred not in good_cred_list:
                    continue
                elif using_cred in good_cred_list:
                    machine_state_list[machine_name_to_index(plan_compromise_machine)]=True
                    may_obtain_cred= self.node_dic[plan_compromise_machine][using_cred]
                    may_obtain_cred_index=[cred_name_to_index(cred_name) for cred_name in may_obtain_cred]

                    obtained_cred_this_machine=[]
                    if using_cred in self.highest_level_credential_list:
                        for index in may_obtain_cred_index:
                            cred_state_list[index]=True
                            obtained_cred_this_machine.append(cred_index_to_name(index))
                    elif using_cred in self.middle_level_credential_list:
                        for index in may_obtain_cred_index:
                            if np.random.rand(1)<1.0:#0.7:
                                cred_state_list[index]=True
                                obtained_cred_this_machine.append(cred_index_to_name(index))
                    elif using_cred in self.lowest_level_credential_list:
                        for index in may_obtain_cred_index:
                            if np.random.rand(1)<1.0:#0.4:
                                cred_state_list[index]=True
                                obtained_cred_this_machine.append(cred_index_to_name(index))
                                
                    self.using_cred_stored[machine_name_to_index(plan_compromise_machine)].append(using_cred)
                    self.obtained_cred[machine_name_to_index(plan_compromise_machine)].extend(copy.deepcopy(obtained_cred_this_machine))

        return machine_state_list,cred_state_list

    def state_transition_temp(self,machine_state_list,cred_state_list,action_contain_list=[]):

        assert (machine_state_list[0]==True or machine_state_list[0]==False)

        if (not (True in machine_state_list)) or (not (True in cred_state_list)):
            return machine_state_list,cred_state_list

        if not action_contain_list == []:
            assert is_number(action_contain_list[0])

        contain_machine_name_list=[machine_index_to_name(index) for index in action_contain_list]
        available_cred= [cred_index_to_name(index) for index in range(len(cred_state_list)) if cred_state_list[index]==True]

        for n in range(len(machine_state_list)):
            if machine_state_list[n]==True and (n not in action_contain_list):
                neighbors_of_n_list=list(self.G.neighbors(self.machine_list[n]))
                neighbors_of_n_list_noncompromised=[machine for machine in neighbors_of_n_list if machine_state_list[machine_name_to_index(machine)]==False]
                potential_plan_compromise_list = [item for item in neighbors_of_n_list_noncompromised]
                if potential_plan_compromise_list==[]:
                    continue

                plan_compromise_machine=random.choice(potential_plan_compromise_list)

                if plan_compromise_machine in contain_machine_name_list:
                    continue

                using_cred=random.choice(available_cred)
                good_cred_list=list(self.node_dic[plan_compromise_machine])
                if using_cred not in good_cred_list:
                    continue
                elif using_cred in good_cred_list:
                    machine_state_list[machine_name_to_index(plan_compromise_machine)]=True
                    may_obtain_cred= self.node_dic[plan_compromise_machine][using_cred]
                    may_obtain_cred_index=[cred_name_to_index(cred_name) for cred_name in may_obtain_cred]

                    obtained_cred_this_machine=[]
                    if using_cred in self.highest_level_credential_list:
                        for index in may_obtain_cred_index:
                            cred_state_list[index]=True
                            obtained_cred_this_machine.append(cred_index_to_name(index))
                    elif using_cred in self.middle_level_credential_list:
                        for index in may_obtain_cred_index:
                            if np.random.rand(1)<1.0:#0.7:
                                cred_state_list[index]=True
                                obtained_cred_this_machine.append(cred_index_to_name(index))
                    elif using_cred in self.lowest_level_credential_list:
                        for index in may_obtain_cred_index:
                            if np.random.rand(1)<1.0:#0.4:
                                cred_state_list[index]=True
                                obtained_cred_this_machine.append(cred_index_to_name(index))

        return machine_state_list,cred_state_list
    
    def reset(self):
        self.obtained_cred=[[] for i in range(self.machine_number)]
        self.using_cred_stored=[[] for i in range(self.machine_number)]

        return

    def set_initial(self,obtained_cred_inital,using_cred_stored_initial):
        self.obtained_cred=obtained_cred_inital
        self.using_cred_stored=using_cred_stored_initial

        return


if __name__ == "__main__":
    my_pomdp=POMDP()

"""
if __name__ == "__main__":
    my_pomdp=POMDP()

    #initial_compro_machine_index=19
    #initial_using_cred_index=1
    initial_compro_machine_index=random.randint(0, my_pomdp.machine_number-1)
    avalible_cred_index=len(list(my_pomdp.node_dic[machine_index_to_name(initial_compro_machine_index)]))
    initial_using_cred_index=random.randint(0, avalible_cred_index-1)
    print(initial_compro_machine_index,initial_using_cred_index)

    my_pomdp.reset()

    machine_state_list=[False]*my_pomdp.machine_number
    machine_state_list[initial_compro_machine_index]=True

    obtained_cred_inital=[[] for i in range(my_pomdp.machine_number)]
    using_cred_stored_initial=[[] for i in range(my_pomdp.machine_number)]
    original_cred_used=list(my_pomdp.node_dic[machine_index_to_name(initial_compro_machine_index)])[initial_using_cred_index]
    using_cred_stored_initial[initial_compro_machine_index]=[original_cred_used]
    original_cred_obtained=my_pomdp.node_dic[machine_index_to_name(initial_compro_machine_index)][original_cred_used]
    obtained_cred_inital[initial_compro_machine_index]=original_cred_obtained
    my_pomdp.set_initial(obtained_cred_inital,using_cred_stored_initial)

    cred_state_list_index=[cred_name_to_index(cred) for cred in original_cred_obtained]
    cred_state_list=[False]*my_pomdp.credential_number
    for index in cred_state_list_index:
        cred_state_list[index]=True

    action_contain_list=[]
    #action_contain_list=[145,361,499,199,246,398,252,102,296,442,155,347,427,320,287]

    for i in range(5000):
        machine_state_list,cred_state_list=my_pomdp.state_transition(machine_state_list,cred_state_list,action_contain_list)
        machine_has_compr=[index for index in range(len(machine_state_list)) if machine_state_list[index]==True]
        machine_has_compr_hop=[my_pomdp.hop[machine_index_to_name(index)] for index in machine_has_compr]
        print("--------------------")
        print(i)
        print(machine_has_compr)
        print(machine_has_compr_hop)

        #print([index for index in range(len(cred_state_list)) if cred_state_list[index]==True]) 
        
        if 0 in machine_has_compr_hop:
            #print([machine_has_compr[j] for j in range(len(machine_has_compr)) if machine_has_compr_hop[j]==1])
            input()
"""
