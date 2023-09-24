from pomdp import *
from model import *

for i in range(175):
    print(i,index_to_action(i))

print(action_to_index([4,5,3]))
print(action_to_index([4,7,5]))
print(action_to_index([9,7,5]))
