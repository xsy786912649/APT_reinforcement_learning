from pomdp import *
from model import *






number=1+18+18*(18-1)/2+18*(18-1)*(18-2)/6-1
number=int(number)
print(number)

for i in range(number+1):
    print(index_to_action(i))