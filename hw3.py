import json
import math
from queue import Queue

#with open('AllReg.json', encoding='utf-8') as file:  # reading json file
 #   ci = json.load(file)  

import json

with open('AllReg.json', encoding="utf8") as f:
    graphjson = json.loads(f.read())
    graph = {}
    graph1 = {}
    vertices_no = 0


def add_vertex(v):
    global graph
    global vertices_no
    vertices_no = vertices_no + 1
    graph[v["name"]] = []
    graph1[v["name"]] = []
# -----------------------------------------------------------------
def add_edge(v1, v2, e):  # Add an edge between vertex v1 and v2 with edge weight e
    global graph

# Since this code is not restricted to a directed or an undirected graph, an edge between v1 v2 does not imply that
# an edge exists between v2 and v1
    temp = [v2, e]
    temp1 = [v2]
    graph[v1["name"]].append(temp)
    graph1[v1["name"]].extend(temp1)
# -----------------------------------------------------------------

for u in graphjson:
    add_vertex(u)

for u1 in graphjson:
    v1 = u1
    for tr in v1["neighbors"]:
        v2 = graphjson[tr["cid"]]["name"]
        e = tr["distance"]
        add_edge(v1, v2, e )



def solcost(route):
    if route == "Start = Goal":
        return 0
    Total = 0
    for (n, distance) in route:
        Total += distance
    return Total

    
def searchId(cityName):  # Take city name and then give you the cid of this cityNme, it take O(n)
    for city in graphjson:
        if city["name"] == cityName:
            return city["cid"]

def searchName(cid):  # Take city id and return city name, it take O(1)
    return graphjson[cid]["name"]


def manhatten(current_cid , goal_cid):
    x = (graphjson[current_cid]["X"] - graphjson[goal_cid]["X"])**2
    y = (graphjson[current_cid]["Y"] - graphjson[goal_cid]["Y"])**2
    return (int)(math.sqrt(x+y)*100)



h_table = {}
def h(goal):
    for city in graphjson:
        h_table[city["cid"]] = manhatten(city["cid"] , goal)



def print_h_cost(path):
    g_cost = 0
    for (node, cost) in path:
        g_cost += cost
    last_node =  searchId(path[-1][0])   
    h_cost =  h_table[last_node] 
    f_cost = h_cost + g_cost
    return h_cost, last_node    


def gready(graph, start, goal):
    visited = []
    queue = [[(start, 0)]]
    gready.num_node = 1
    gready.max_fringe = 1
    if start == goal:
        return 'No solution'
    while queue:
        queue.sort(key= print_h_cost)  # sort by cost
        path = queue.pop(0)
        node = path[-1][0]
        if node not in visited:
            adj_nodes = graph[node]
            gready.num_node += len(adj_nodes)
            if len(queue)+len(adj_nodes) > gready.max_fringe:
                gready.max_fringe = len(queue)+len(adj_nodes)
            visited.append(node)
            if node == goal:
                return path
            else:
                for (node2, cost) in adj_nodes:
                    new_path = path.copy()
                    new_path.append((node2, cost))
                    queue.append(new_path)



def path_cost(path):
    total_cost = 0
    for (node, cost) in path:
        total_cost += cost
    return total_cost, path[-1][0]


#-------------------------------------------------------------------------------------------------------------------------------------
def print_f_cost(path):
    g_cost = 0
    for (node, cost) in path:
        g_cost += cost
    last_node =  searchId(path[-1][0])   
    h_cost =  h_table[last_node] 
    f_cost = h_cost + g_cost
    return f_cost, path[-1][0]    

def A_search(graph, start, goal):
    #visited = []
    queue = [[(start, 0)]]
    A_search.gen_node = 1
    A_search.num_fringe = 1
    if start == goal:
        return 'No solution'
    while queue:
        queue.sort(key= print_f_cost)  # sort by cost
        path = queue.pop(0)
        node = path[-1][0]
        #if node not in visited:
        adj_nodes = graph[node]
        A_search.gen_node += len(adj_nodes)
        if len(queue)+len(adj_nodes) > A_search.num_fringe:
            A_search.num_fringe = len(queue)+len(adj_nodes)
            #visited.append(node)
        if node == goal:
            return path
        else:
            for (node2, cost) in adj_nodes:
                new_path = path.copy()
                new_path.append((node2, cost))
                queue.append(new_path)





initial_state_name = "طريف"
goal_state_name = "حائل"
goal = searchId(goal_state_name)
h(goal)

gas = input("Enter the KM/liter of gas: ")




print("---------------------------------------------------------------")
print("\n Gready Search : ")
sol = gready(graph,initial_state_name,goal_state_name)
count = 0
print("The cost is = {:.2f}".format((path_cost(sol)[0]/float(gas)) * 2.18))
print("Route =  ",end= ' ')
lis = []
for i in sol:
   #print(sol[count][0], end=',')
   lis.append(sol[count][0])
   count += 1
print(lis)
print("Distance = ", path_cost(sol)[0])
print("number of generated nodes =  ", gready.num_node)
print("Maximum fringe size = ", gready.max_fringe)



print("---------------------------------------------------------------")
print("\n A* Search : ")
sol = A_search(graph,initial_state_name,goal_state_name)
count = 0
print("The cost is = {:.2f}".format((path_cost(sol)[0]/float(gas)) * 2.18))
print("Route =  ",end= ' ')
lis = []
for i in sol:
   #print(sol[count][0], end=',')
   lis.append(sol[count][0])
   count += 1
print(lis)
print("Distance = ", path_cost(sol)[0])
print("number of generated nodes =  ",A_search.gen_node)
print("Maximum fringe size = ", A_search.num_fringe)