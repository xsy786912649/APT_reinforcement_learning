from pomdp import *
from model import *

n=12
for i in range(int(1+n+n*(n-1)/2)+1):
    print(index_to_action(i))