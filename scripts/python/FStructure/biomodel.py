#!/usr/bin/python
import json

treename = "biomodel"
fdim = 2
xdim = 2
constant_range = [0 ,10]
randseed = 1

complexity = 4

def is_N(n,N):
    if n == N:
        return 1.0
    else:
        return 0.0

forest = []
for i in range(fdim):
    q = 0
    rootid = i * 10000
    tree = {"tree" : rootid, "nodes":[]}
    tree["nodes"].append({"type":"CT_MINUS","node_id":rootid,"children":[rootid+1,rootid+2],"weights":[],"trainable":False})
    tree["nodes"].append({"type":"*","node_id":rootid+1,"children":complexity,"weights":[],"trainable":True})
    tree["nodes"].append({"type":"CT_MUL2","node_id":rootid+2,"children":[rootid+3,rootid+4],"weights":[],"trainable":False})
    tree["nodes"].append({"type":"CT_VAR","node_id":rootid+3,"children":[],"weights":[is_N(i,fdim) for i in range(1,fdim+1)],"trainable":False})
    tree["nodes"].append({"type":"*","node_id":rootid+4,"children":complexity,"weights":[],"trainable":True})
    forest.append(tree)



biomodel = {"treename":treename,"fdim":fdim,"xdim":xdim,"constant_range":constant_range,"randseed":randseed,"forest":forest}
print(json.dumps(biomodel,indent=4, separators=(',', ': ')))
