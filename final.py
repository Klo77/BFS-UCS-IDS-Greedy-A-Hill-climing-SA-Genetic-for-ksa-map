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

def solcost(route):
    if route == "Start = Goal":
        return 0
    Total = 0
    for (n, distance) in route:
        Total += distance
    return Total


for u in graphjson:
    add_vertex(u)

for u1 in graphjson:
    v1 = u1
    for tr in v1["neighbors"]:
        v2 = graphjson[tr["cid"]]["name"]
        e = tr["distance"]
        add_edge(v1, v2, e )



def ucs(graph, start, goal):
    visited = []
    queue = [[(start, 0)]]
    ucs.num_node = 1
    ucs.max_fringe = 1
    if start == goal:
        return 'No solution'
    while queue:
        queue.sort(key=path_cost)  # sort by cost
        path = queue.pop(0)
        node = path[-1][0]
        if node not in visited:
            adj_nodes = graph[node]
            ucs.num_node += len(adj_nodes)
            if len(queue)+len(adj_nodes) > ucs.max_fringe:
                ucs.max_fringe = len(queue)+len(adj_nodes)
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


path = []
pathCost = []
newpath = []
visited2 = []
maxfringe = []
global uuu
uuu = 0
def DFS(start, goal, graph, maxDepth, queue, j):
   # print("Checking for destination", start)
    queue.append(start)
    global uuu
    uuu +=0
    if start == goal:
        path.append(queue)
        newpath.append(pathCost)
        DFS.maxfringe = j
        #print("Maximum fringe size: ", j)
        return True
    if maxDepth <= 0:
        path.append(queue)
        newpath.append(pathCost)
        return False
    for node in graph[start]:
        pathCost.append(node[1])
        if len(graph[start])+len(queue)+len(pathCost) > uuu:
            uuu = len(graph[start])+len(queue)+len(pathCost)
        if node not in visited2:
            visited2.append(node)
            if len(graph[start])+len(queue)+len(pathCost) > uuu:
                uuu = len(graph[start])+len(queue)+len(pathCost)
        if DFS(node[0], goal, graph, maxDepth - 1, queue, j):
            return True
        else:
            pathCost.pop()
            queue.pop()
    return False


def iterativeDDFS(start, goal, graph, maxDepth):
    for i in range(maxDepth):
        queue = []
        if DFS(start, goal, graph, i, queue,0):
            return True
    return False



import json
from queue import Queue

with open('AllReg.json', encoding='utf-8') as file:  # reading json file
    ci = json.load(file)


class node:  # node is the city in the queue
    def __init__(self, cid, distance, parent=None):
        self.cid = cid
        self.distance = distance
        self.parent = parent  # The parent of current city

    def getCid(self):
        return self.cid

    def getDistance(self):
        return self.distance

    def __str__(self):  # Just for testing
        if self is not None:
            return f"[{self.cid}]: dicetance: [{self.distance}], parent: {self.parent}"  # if you print __str__ for current city it will print all the parents


def makeNode(cid, distance, parent):  # To make new city
    return node(cid, distance, parent)


def searchId(cityName):  # Take city name and then give you the cid of this cityNme, it take O(n)
    for city in ci:
        if city["name"] == cityName:
            return city["cid"]


def searchName(cid):  # Take city id and return city name, it take O(1)
    return ci[cid]["name"]


def goalFunction(goal, city):  # Take goal: The fiale city, and city: the current city
    return city == goal  # And retuen ture if the current city is the goal, false otherwise


visited = {}  # For every city in visited if it is true it mean, it is already visited, if not visited it will be false
for city in ci:
    visited[city["cid"]] = False

fringe = Queue()
maxFringeSize = fringe.qsize()


def calcMaxFringeSize():
    global maxFringeSize
    if maxFringeSize < fringe.qsize():
        maxFringeSize = fringe.qsize()



initial_state_name = "نجران"
initial_state_cid = searchId(initial_state_name)
visited[initial_state_name] = True  # At the start we will put the initial state in the finge
fringe.put(makeNode(initial_state_cid, 0, None))
global nNodes
nNodes = 1

goal_state_name = "سكاكا"
goal_state_cid = searchId(goal_state_name)



def expand(id, parent):  # Take id and parent of the current city then it finds the successors and puts them in the fringe
    listOfNeighbors = ci[id]["neighbors"]
    for i in listOfNeighbors:
        if not visited[i['cid']]:
            fringe.put(makeNode(i['cid'], i['distance'], parent))
            global nNodes
            nNodes += 1
    calcMaxFringeSize()


# BFS
def bfs():  # Return the final city as node
    while not fringe.empty():
        temp = fringe.get()
        visited[temp.getCid()] = True
        if goalFunction(goal_state_cid, temp.getCid()):
            return temp
        expand(temp.getCid(), temp)





def print_path(p):
    print_path.path = []
    print_path.distance = 0
    while p is not None:
        print_path.path.append(searchName(p.getCid()))
        print_path.distance += p.getDistance()
        p = p.parent
    print_path.path.reverse()
    print("-----------------------------------------------------------------------------------------------")
    print("BFS: ")
    print(f"Route : {print_path.path}")
    print(f"Distance = {print_path.distance} km")



x = bfs()
# testing
print_path(x)
print(f"number of generated nodes = {nNodes}")
print(f"Maximum fringe size =  {maxFringeSize}")


print("---------------------------------------------------------------")
print("\nUCS: ")
sol = ucs(graph,initial_state_name,goal_state_name)
count = 0
print("Route =  ",end= ' ')
lis = []
for i in sol:
   #print(sol[count][0], end=',')
   lis.append(sol[count][0])
   count += 1
print(lis)
print("Distance = ", path_cost(sol)[0])
print("number of generated nodes = = ", ucs.num_node)
print("Maximum fringe size = ", ucs.max_fringe)
    

print("\n---------------------------------------------------------------")
print("IDS: ")
if iterativeDDFS(initial_state_name, goal_state_name, graph,100):
        #print(".......................................................")
        print("Route = ",path.pop())
        c = len(newpath[0])
        s = 0
        for i in range(0,c):

            s+= newpath[i][i]
        print("Distance =   ",s)
        print("Number of generated nodes = ", len(visited2))
        print("Maximum fringe size = ", uuu)
else:
     print(".......................................................")
     print("Goal Path is not available")
     print(".......................................................")

print("---------------------------------------------------------------")