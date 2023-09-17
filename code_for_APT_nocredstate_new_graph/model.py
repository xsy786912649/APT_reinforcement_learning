import random
import numpy as np
import pickle
from pomdp import *
import os
import sys

with open(f'./APT_data/hop.pickle','rb') as f:
    P0=pickle.load(f)
N_hop=[[] for i in range(7)]
for index_machine in list(P0):
    N_hop[int(P0[index_machine])].append(index_machine)
target=N_hop[0]
hop_1=N_hop[1]
hop_2=N_hop[2]
hop_3=N_hop[3]
contain_hop=hop_1+hop_2
my_pomdp1=POMDP()

eps=0.3
lr=0.5

def index_to_action(index):
    if index==0:
        return []
    elif index>=1 and index<=14:
        return [machine_name_to_index(contain_hop[index-1])]
    elif index>14:
        if index-14<=13:
            a1=0
            b1=index-14
        elif index-14-13<=12:
            a1=1
            b1=index-14-13+a1
        elif index-14-13-12<=11:
            a1=2
            b1=index-14-13-12+a1
        elif index-14-13-12-11<=10:
            a1=3
            b1=index-14-13-12-11+a1
        elif index-14-13-12-11-10<=9:
            a1=4
            b1=index-14-13-12-11-10+a1
        elif index-14-13-12-11-10-9<=8:
            a1=5
            b1=index-14-13-12-11-10-9+a1
        elif index-14-13-12-11-10-9-8<=7:
            a1=6
            b1=index-14-13-12-11-10-9-8+a1
        elif index-14-13-12-11-10-9-8-7<=6:
            a1=7
            b1=index-14-13-12-11-10-9-8-7+a1
        elif index-14-13-12-11-10-9-8-7-6<=5:
            a1=8
            b1=index-14-13-12-11-10-9-8-7-6+a1
        elif index-14-13-12-11-10-9-8-7-6-5<=4:
            a1=9
            b1=index-14-13-12-11-10-9-8-7-6-5+a1
        elif index-14-13-12-11-10-9-8-7-6-5-4<=3:
            a1=10
            b1=index-14-13-12-11-10-9-8-7-6-5-4+a1
        elif index-14-13-12-11-10-9-8-7-6-5-4-3<=2:
            a1=11
            b1=index-14-13-12-11-10-9-8-7-6-5-4-3+a1
        elif index-14-13-12-11-10-9-8-7-6-5-4-3-2<=1:
            a1=12
            b1=index-14-13-12-11-10-9-8-7-6-5-4-3-2+a1
        a=machine_name_to_index(contain_hop[a1])
        b=machine_name_to_index(contain_hop[b1])
        return [a,b]

def action_to_index(action):
    assert (action==[] or (machine_index_to_name(action[0]) in contain_hop))
    if action==[]:
        return 0
    elif len(action)==1:
        return contain_hop.index(machine_index_to_name(action[0]))+1
    elif len(action)==2:
        assert (machine_index_to_name(action[0]) in contain_hop) and (machine_index_to_name(action[1]) in contain_hop)
        a=contain_hop.index(machine_index_to_name(action[0]))
        b=contain_hop.index(machine_index_to_name(action[1]))
        if a<b:
            return int(a*(25-a)/2+b+14) 
        else: 
            return int(b*(25-b)/2+a+14) 

def full_state_to_simplest_state(machine_state_list):
    machine_has_compr_name=[machine_index_to_name(index) for index in range(len(machine_state_list)) if machine_state_list[index]==True] 
    machine_simplest_state=[(contain_hop[i] in machine_has_compr_name) for i in range(len(contain_hop))]

    return machine_simplest_state

def full_state_to_higher_state(machine_state_list):
    attention_hop=hop_1+hop_2+hop_3
    machine_has_compr_name=[machine_index_to_name(index) for index in range(len(machine_state_list)) if machine_state_list[index]==True] 
    machine_higher_state=[(attention_hop[i] in machine_has_compr_name) for i in range(len(attention_hop))]

    return machine_higher_state

def simplest_state_to_valuedic_key(machine_simplest_state):
    state=machine_simplest_state
    valuedic_key=0
    for i in range(len(state)):
        if state[i]==True:
            valuedic_key+=int(pow(2,i))
    return valuedic_key

def random_attacker_start(my_pomdp, seed=None) :
    if not seed == None:
        random.seed(seed)
        np.random.seed(seed)

    my_pomdp.reset()
    initial_compro_machine_index=random.randint(0, my_pomdp.machine_number-1)
    while machine_index_to_name(initial_compro_machine_index) in N_hop[0]+N_hop[1]:
        initial_compro_machine_index=random.randint(0, my_pomdp.machine_number-1)
    avalible_cred_index=len(list(my_pomdp.node_dic[machine_index_to_name(initial_compro_machine_index)]))
    initial_using_cred_index=random.randint(0, avalible_cred_index-1)
    #print(initial_compro_machine_index,initial_using_cred_index)

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

    machine_state_list_belief_prability=np.array([0.03 for i in range(len(machine_state_list))])
    cred_state_list_belief_prability=np.array([0.03 for i in range(len(cred_state_list))])

    random.seed(None)
    np.random.seed(None)

    return machine_state_list,cred_state_list,machine_state_list_belief_prability,cred_state_list_belief_prability 

if __name__ == "__main__":
    weight = float(sys.argv[1]) #10.0
    base_penalty= float(sys.argv[2])
    if os.path.exists("./model_"+str(weight)+".pkl"):
        with open(f"./model_"+str(weight)+".pkl",'rb') as f:
            value_map_dict=pickle.load(f)
    else:
        value_map_dict = {}
        for i in range(pow(2,14)):
            value_map_dict[i]=[]
            value_map_dict[i].append(-2000.0)
            for j in range(14):
                value_map_dict[i].append(-2000.0)
                for k in range(j):
                    value_map_dict[i].append(-2000.0)

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
            simplest_state_current_machine=full_state_to_simplest_state(machine_state_list)
            current_valuedic_key=simplest_state_to_valuedic_key(simplest_state_current_machine)
            Q_value_current=value_map_dict[current_valuedic_key]

            if no_change_action==False:
                #action_index=0
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

            if ((0 in machine_has_compr_hop_new) or (1 in machine_has_compr_hop_new) or (2 in machine_has_compr_hop_new)): 
                simplest_state_new_machine=full_state_to_simplest_state(machine_state_list_new)
                new_valuedic_key=simplest_state_to_valuedic_key(simplest_state_new_machine)
                Q_value_new=value_map_dict[new_valuedic_key]

                reward_safe=0.0
                if 0 in machine_has_compr_hop_new:
                    reward_safe=-1000.0
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
                    value_map_dict[current_valuedic_key][action_index]=value_map_dict[current_valuedic_key][action_index]*(1-lr)+lr*(reward+1999.0/2000*max(Q_value_new))
                else:
                    value_map_dict[current_valuedic_key][action_index]=value_map_dict[current_valuedic_key][action_index]*(1-lr)+lr*(reward-2000.0)

                print(simplest_state_new_machine)
                print(value_map_dict[current_valuedic_key][action_index])
            
            if full_state_to_simplest_state(machine_state_list)==full_state_to_simplest_state(machine_state_list_new):
                no_change_action=True
            else:
                no_change_action=False

            machine_state_list=machine_state_list_new
            cred_state_list=cred_state_list_new
            print(action_contain_list)
            print(i) 

            if 0 in machine_has_compr_hop_new:
                print(value_map_dict[current_valuedic_key])
                break

    f_save=open("model_"+str(weight)+".pkl",'wb')
    pickle.dump(value_map_dict,f_save)
    f_save.close()
    
            
        

        
        
            






