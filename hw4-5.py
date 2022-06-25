import json
from os import name
import random
import math
from math import radians,cos,sin,asin,sqrt
import copy
from re import S
from turtle import color
import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text
import operator

with open('AllReg.json', encoding="utf8") as f:
    graphjson = json.loads(f.read())

def searchId(cityName):  # Take city name and then give you the cid of this cityNme, it take O(n)
    for city in graphjson:
        if city["name"] == cityName:
            return city["cid"]

def searchName(cid):  # Take city id and return city name, it take O(1)
    return graphjson[cid]["name"]

def searchX(cid):
    return graphjson[cid]["X"]

def searchY(cid):
    return graphjson[cid]["Y"]



def manhatten(current_cid , goal_cid):
    x = (graphjson[current_cid]["X"] - graphjson[goal_cid]["X"])**2
    y = (graphjson[current_cid]["Y"] - graphjson[goal_cid]["Y"])**2
    return (int)(math.sqrt(x+y)*100)


def finding_straight_line_distance(cid1, cid2):
    x1 = radians(graphjson[cid1]["X"])
    y1 = radians(graphjson[cid1]["Y"])
    x2 = radians(graphjson[cid2]["X"])
    y2 = radians(graphjson[cid2]["Y"])
    # Haversine formula
    dlon = y2 - y1
    dlat = x2 - x1
    a = sin(dlat / 2) ** 2 + cos(x1) * cos(x2) * sin(dlon / 2) ** 2

    c = 2 * asin(sqrt(a))

    # Radius of earth in kilometers. Use 3956 for miles
    r = 6371

    # calculate the result
    return (int)(c * r)

WIDTH = 640
HEIGHT = 480




number = input("Enter the number of cities: ")
number = (int)(number)


cities_name = list()
counter = 1
while number > 0:
    city = input(str(counter)+":  ")
    cities_name.append(city)
    number = number - 1
    counter+= 1



cities_id = list()
for i in cities_name:
    cities_id.append(searchId(i))


x_value = [round(searchX(i),4) for i in cities_id]
y_value = [round(searchY(i),4) for i in cities_id]


i = 0
tsp = [[]]

while i < len(cities_id):
    j = 0
    while j < len(cities_id) :
        tsp[i].append(finding_straight_line_distance(cities_id[i],cities_id[j]))
        j = j+1  
    i = i + 1 
    if(i < len(cities_id)):
        tsp.append(list()) 

def convert_to_city_names(city_ids):
    city_names = list()
    for i in city_ids:
        city_names.append(searchName(i))
    return city_names    


it = (int)(input("Enter the Iteration: "))
current_iteration = 1

gas = input("Enter the KM/liter of gas: ")


xcord = [round(searchX(i),4) for i in cities_id]
ycord = [round(searchY(i),4) for i in cities_id]

def updateGraph(path,routeLength,temperature,name):
    plt.cla()
    plt.scatter(xcord ,ycord,color ='red')
    x = []
    y = []
    for i in path:
        x.append(xcord[i])
        y.append(ycord[i])
    x.append(xcord[0])
    y.append(ycord[0])
    if(name == 'h_c'):
        title = f'Hell Climing:  \n'
    if name == 's_a':
        title = f'Simulated aneeling\n'
        title = title +'         '+f'temperature = {temperature:.3f}'
    if name == 'g_a':
         title = f'Genetic : \n'
    title = title + '       ' + f'best cost: {routeLength:.3f}'
    plt.title(title)
    names = [searchName(cities_id[i]) for i in range(len(cities_id))] 
    texts = [plt.text(xcord[i],ycord[i],f'N : {reverse_string(names[i]) }') for i in range(len(names))]
    adjust_text(texts)
    plt.plot(x,y,color='blue')
    plt.pause(1)

def reverse_string(name):
    reverse_s = ""
    for i in name:
        reverse_s = i+reverse_s
    return reverse_s


def plot_pop(cities,distance):
    plt.cla()
    x = [i[0] for i in cities]
    y = [i[1] for i in cities]
    x.append(cities[0][0])
    y.append(cities[0][1])
    plt.scatter (xcord, ycord, color = 'red')
    plt.plot(x,y,color='blue')
    title = f'Genetic : \n'
    title = title + '       ' + f'best cost: {distance:.3f}'
    plt.title(title)
    names = [searchName(cities_id[i]) for i in range(len(cities_id))] 

    texts = [plt.text(xcord[i],ycord[i],f'N : {reverse_string(names[i]) }') for i in range(len(names))]
    adjust_text(texts)
    plt.plot(x,y,color='blue')
    plt.pause(1)


def random_solution(tsp, cities_id):
    cities = list(range(len(tsp)))
    solution = []
    for i in range(len(tsp)):
        random_city = cities[random.randint(0,len(cities)-1)]
        solution.append(random_city)
        cities.remove(random_city)
    return solution


def routeLength(tsp, solution):
    routeLength = 0 
    for i in range(len(solution)):
        routeLength += tsp[solution[i-1]][solution[i]]
    return routeLength    

def convert_index_to_solution(solution):
    sol = list()
    for i in range(0,len(solution)):
        sol.append(cities_id[solution[i]])
    return sol    


def convert_solution_to_index(ids):
    for i in range(len(cities_id)):
        ids[i] = cities_id[ids[i]]
    return ids

def getNeighbors(solution):
    neighbors = []
    for i in range(len(solution)):
        for j in range(i+1 , len(solution)):
            neighbor = solution.copy()
            neighbor[i] = solution[j]
            neighbor[j] = solution[i]
            neighbors.append(neighbor)
    return neighbors        


def getBestNeighbors(tsp , neighbors):
    ids = convert_index_to_solution(neighbors[0])
    bestRouteLength = calculateDistance(ids)
    bestNeighbor = neighbors[0]
    for neighbor in neighbors:
        ids = convert_index_to_solution(neighbor)
        currentRouteLength = calculateDistance(ids)
        if currentRouteLength < bestRouteLength:
            bestRouteLength = currentRouteLength
            bestNeighbor = neighbor

    return bestNeighbor , bestRouteLength        

def h_c(tsp,cities_id,iteration):
    current_iteration = 1
    current_solution = random_solution(tsp,cities_id)
    ids = convert_index_to_solution(current_solution)
    currentRouteLength = calculateDistance(ids)
    updateGraph(current_solution,currentRouteLength,0,'h_c')
    neighbors = getNeighbors(current_solution)
    bestNeighbor , bestNeighborRouteLength = getBestNeighbors(tsp , neighbors)
   
    while bestNeighborRouteLength < currentRouteLength:
        current_solution = bestNeighbor
        currentRouteLength = bestNeighborRouteLength
        if iteration == current_iteration:
            now = convert_index_to_solution(current_solution)
            path_now = convert_to_city_names(now)
            print("route at iteration "+str(iteration)+" is "+str(path_now))
            print("distance at  iteration  "+str(iteration)+" = {:.2f}".format(currentRouteLength))
            print("The cost is =  "+str(current_iteration)+" = {:.2f}".format((currentRouteLength/float(gas)) * 2.18)+" SR")
            print("")
        current_iteration+=1
        neighbors = getNeighbors(current_solution)
        bestNeighbor , bestNeighborRouteLength = getBestNeighbors(tsp , neighbors)
        #print("distance in H_C = {:.2f}".format(currentRouteLength))
        updateGraph(current_solution,currentRouteLength,0,'h_c')
    solution_path_id = convert_index_to_solution(current_solution)
    solution_path_names = convert_to_city_names(solution_path_id)
    plt.show()
    return solution_path_names , currentRouteLength

#-------------------------------------------------------------------------------------------------------------------------------------------------



def acceptanceProbability(bestRouteLength , new_route_length , temprature):
    if(new_route_length < bestRouteLength):
        return 1
    return math.exp((bestRouteLength - new_route_length)/temprature)

    

def S_A(tsp , cities_id,iteration):
    current_iteration = 1
    temprature = 1000
    cooling_rate = 0.4
    current_solution = random_solution(tsp,cities_id)
    currentRouteLength = calculateDistance(current_solution)
    #bestSolution = random_solution(tsp,cities_id)
    #bestRouteLength = routeLength(tsp,bestSolution)
    best_distance = []
    while temprature > 1: 
        neighbors = getNeighbors(current_solution)
        bestNeighbor , bestNeighborRouteLength = getBestNeighbors(tsp , neighbors)
       # new_solution = bestSolution
        #random_index1 = random.randint(0,len(bestSolution)-1)
        #random_index2 = random.randint(0,len(bestSolution)-1)

        #new_solution[random_index1] , new_solution[random_index2] = bestSolution[random_index2],bestSolution[random_index1]
        #new_route_length = routeLength(tsp,new_solution)
        #if(acceptanceProbability(bestRouteLength , new_route_length , temprature) > random.random()):
        if(acceptanceProbability(currentRouteLength , bestNeighborRouteLength , temprature) > random.random()):
            current_solution = bestNeighbor
            currentRouteLength = bestNeighborRouteLength
            updateGraph(current_solution,currentRouteLength,temprature,'s_a')
            current_iteration+=1
            neighbors = getNeighbors(current_solution)
            bestNeighbor , bestNeighborRouteLength = getBestNeighbors(tsp , neighbors)
            best_distance.append(bestNeighborRouteLength)
            if iteration == current_iteration:
                now = convert_index_to_solution(current_solution)
                path_now = convert_to_city_names(now)
                print("route at iteration "+str(current_iteration)+" is "+str(path_now))
                print("distance at  iteration  "+str(current_iteration)+" = {:.2f}".format(currentRouteLength)+" Km")
                print("The cost at iteration "+str(current_iteration)+" = {:.2f}".format((currentRouteLength/float(gas)) * 2.18)+" SR")

                print("")  

            #bestSolution = new_solution
            #bestRouteLength = new_route_length
        
        if(currentRouteLength < bestNeighborRouteLength):
            #bestSolution = new_solution
            #bestRouteLength = new_route_length  
            current_solution = bestNeighbor
            currentRouteLength = bestNeighborRouteLength  
            updateGraph(current_solution,currentRouteLength,temprature,'s_a')

            current_iteration+=1
            neighbors = getNeighbors(current_solution)
            bestNeighbor , bestNeighborRouteLength = getBestNeighbors(tsp , neighbors) 
            best_distance.append(bestNeighborRouteLength)
            if iteration == current_iteration:
                now = convert_index_to_solution(current_solution)
                path_now = convert_to_city_names(now)
                print("route at iteration "+str(current_iteration)+" is "+str(path_now))
                print("distance at  iteration  "+str(current_iteration)+" = {:.2f}".format(currentRouteLength)+" Km")
                print("The cost at iteration "+str(current_iteration)+" = {:.2f}".format((currentRouteLength/float(gas)) * 2.18)+" SR")

                print("")
       # print(bestRouteLength) 
        #print("Distance in S_A = {:.2f}".format(currentRouteLength))   
        temprature *= cooling_rate   

    #solution_path_id = convert_solution(bestSolution)
    #solution_path_names = convert_to_city_names(solution_path_id)
    #return solution_path_names , bestRouteLength   
    solution_path_id = convert_index_to_solution(current_solution)
    solution_path_names = convert_to_city_names(solution_path_id)
    plt.show()
    return solution_path_names , min(best_distance)   


#-------------------------------------------------------------------------------------------------------------------------------------------------


def calculateDistance(population_id):
    sum = 0
    for i in range(len(population_id)-1):
        sum+= finding_straight_line_distance(population_id[i],population_id[i+1])
    sum+= finding_straight_line_distance(population_id[0],population_id[-1])
    return sum    



print("---------------------------------")
print("")
final_HC = h_c(tsp,cities_id,it)
final_solution_cities_HC = final_HC[0]
final_solution_distance_HC = final_HC[1]
print("final route in H_C = "+str(final_solution_cities_HC))
print("final distance in H_C =  "+str(final_solution_distance_HC)+" Km")
print("The cost in H_C  = {:.2f}".format((final_solution_distance_HC/float(gas)) * 2.18)+" SR")



print("")
print("---------------------------------")
print("")
final_SA = S_A(tsp,cities_id,it)
final_solution_cities_SA = final_SA[0]
final_solution_distance_SA = final_SA[1]
print("final route in S_A  = "+str(final_solution_cities_SA))
print("final distancein S_A = "+str(final_solution_distance_SA)+" Km")
print("The cost in S_A  = {:.2f}".format((final_solution_distance_SA/float(gas)) * 2.18)+" SR")


print("")
print("---------------------------------")
print("")


#-------------------------------------------------------------------------------------------------------------------------------------------------

cityList  = [[]]
i = 0 
for i in range(len(cities_id)):
    cityList[i].append(round(searchX(cities_id[i]),4))
    cityList[i].append(round(searchY(cities_id[i]),4))
    if i == len(cities_id)-1:
        break
    cityList.append(list())



def distance_between_cities(cities):
    data = dict()
    for index, value in enumerate(cities):
        x1 = cities[index][0]
        y1 = cities[index][1]
        if index + 1 <= len(cities)-1:
            x2 = cities[index+1][0]
            y2 = cities[index+1][1]
            xdiff = x2 - x1
            ydiff = y2 - y1
            dst = (xdiff*xdiff + ydiff*ydiff)** 0.5
            data['Distance from city '+ str(index+1) +' to city ' + str(index+2)] = dst 
        elif index + 1 > len(cities)-1:
            x2 = cities[0][0]
            y2 = cities[0][1]
            xdiff = x2 - x1
            ydiff = y2 - y1
            dst = (xdiff*xdiff + ydiff*ydiff)** 0.5
            #dst = finding_straight_line_distance(cities_id[0],cities_id[-1])
            data['Distance from city '+ str(index+1) + ' to city ' + str(index +2 -len(cities))] = dst
              
    return data



def total_distance(cities):
    total = sum(distance_between_cities(cities).values())
    return total


def generatePath(cities):
    path = random.sample(cities, len(cities))
    return path

def initialPopulation(cities, populationSize):
    population = [generatePath(cities) for i in range(0, populationSize)]
    return population



def path_fitness(cities):
    total_dis = total_distance(cities)
    fitness= 0.0
    if fitness == 0:
        fitness = 1 / float(total_dis)
    return fitness

def get_names(result_lst, cities, name_lst):
    names = []
    for index,value in enumerate(result_lst):
        for i,v in enumerate(cities):
            if value == v:
                names.append(name_lst[i])
    return names

def rankPathes(population):
    fitnessResults = {}
    for i in range(len(population)):
            #sol = get_names(t,cityList , cities_name)
            #print("route at iteration "+str(it)+" is ",end =" ")
            #print([(val) for indx,val in enumerate(sol)])
            #ids = []
            #for val in enumerate(sol):
             #  ids.append(searchId(val[1]))
            #distance  = 0
            #for index in range(len(ids)-1):
             #   distance +=  finding_straight_line_distance(ids[index],ids[index+1])
            #print("distance at  iteration  "+str(it)+" = {:.2f}".format(distance))
        fitnessResults[i] = path_fitness(population[i])
        
    return sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = True)



def perform_selection(pop, eliteSize):
    #output = rankPathes(population)
#A cumulative sum is a sequence of partial sums of a given sequence
#Cumulative percentage is another way of expressing frequency distribution. 
#It calculates the percentage of the cumulative frequency within each interval, much as relative frequency distribution calculates the percentage of frequency.
    selected_values = [pop[i][0] for i in range(eliteSize)]
    
    for i in range(len(pop) - eliteSize):
        pick = 100*random.random()
        for i in range(0, len(pop)):
                selected_values.append(pop[i][0])
                break
                
    return selected_values




def do_mating_pool(population, selected_values):
    matingpool = [population[selected_values[i]] for i in range(len(selected_values))]
    return matingpool


def do_breed(first_parent, second_parent):
    generation_1= int(random.random() * len(first_parent))
    generation_2 = int(random.random() * len(second_parent))
    
    first_generation = min(generation_1, generation_2)
    last_generation = max(generation_1, generation_2)

    tot_parent1 = [first_parent[i] for i in range(first_generation, last_generation)]
    tot_parent2 = [i for i in second_parent if i not in tot_parent1]

    tot = tot_parent1 + tot_parent2
    return tot

def do_breed_population(my_mating_pool, eliteSize):
    ln = len(my_mating_pool) - eliteSize
    pl = random.sample(my_mating_pool, len(my_mating_pool))
    tot1 = [my_mating_pool[i] for i in range(eliteSize)]
    tot2 = [do_breed(pl[i], pl[len(my_mating_pool)-i-1]) for i in range(ln)]
    tot = tot1+tot2
    return tot



def do_mutatation(indiv, mutat_rate):
    for exchanged in range(len(indiv)):
        if(random.random() < mutat_rate):
            exchanged_with = int(random.random() * len(indiv))
            
            city1 = indiv[exchanged]
            city2 = indiv[exchanged_with]
            
            indiv[exchanged] = city2
            indiv[exchanged_with] = city1
    return indiv

def do_mutatation_pop(population, mutat_rate):
    mutated_population = [do_mutatation(population[i], mutat_rate) for i in range(len(population))]
    return mutated_population


def get_following_gen(existing_gen, eliteSize, mutat_rate):
    pop = rankPathes(existing_gen)
    
    selected_values = perform_selection(pop, eliteSize)
   
    my_mating_pool = do_mating_pool(existing_gen, selected_values)
    tot = do_breed_population(my_mating_pool, eliteSize)
    following_gen = do_mutatation(tot, mutat_rate)
    #print(following_gen)
    return following_gen








def GA(city_names,cities,it, population_size, eliteSize, mutat_rate, generations):
    population = initialPopulation(cities,population_size)
    #print(population_)
 
    for i in range(generations):

        population = get_following_gen(population, eliteSize, mutat_rate)
        #print(population)
    

    current = 1
    population.reverse()
    c = len(population)-1
    for i in range(len(population)):
        if it == current:
            sol = rankPathes(population)[c][0]
            route = population[sol]
            name = get_names(route,cities,cities_name)
            print("route at iteration "+str(it)+" is ",end =" ")
            print(name)
            ids_it = []
            for  i in range(len(name)):
                ids_it.append(searchId(name[i]))
            dist_it = 0
            for i in range(len(ids_it)-1):
                dist_it +=  finding_straight_line_distance(ids_it[i],ids_it[i+1])
            dist_it += finding_straight_line_distance(ids_it[0],ids_it[-1])
            print("final distance at iteration "+str(it)+" in G_E = " + str(dist_it)+" Km")
            print("The cost at iteration "+str(it)+" in G_E  = {:.2f}".format((dist_it/float(gas)) * 2.18)+" SR")

        current+=1
        c-= 1

    c = len(population) - 1
    for i in range(len(population)):
        sol = rankPathes(population)[c][0]
        route = population[sol]
        name = get_names(route,cities,cities_name)
        ids_it = []
        for  i in range(len(name)):
            ids_it.append(searchId(name[i]))
        dist_it = 0
        for i in range(len(ids_it)-1):
            dist_it +=  finding_straight_line_distance(ids_it[i],ids_it[i+1])
        dist_it += finding_straight_line_distance(ids_it[0],ids_it[-1])
        plot_pop(route,dist_it)
        c-= 1

    population.reverse()
    print()
    optimal_route_id = rankPathes(population)[0][0]
    optimal_route = population[optimal_route_id]
    ordered_cities = get_names(optimal_route,cities,cities_name)

    print("final route in G_E =" ,end =" ")
    print(ordered_cities)
    ids = []
    for  i in range(len(ordered_cities)):
        ids.append(searchId(ordered_cities[i]))
    dest = 0
    for i in range(len(ids)-1):
        dest +=  finding_straight_line_distance(ids[i],ids[i+1])
    dest += finding_straight_line_distance(ids[0],ids[-1])
    print("final distance in G_E = " + str(dest)+" Km")
    print("The cost in G_E  = {:.2f}".format((dest/float(gas)) * 2.18)+" SR")

        
    plot_pop(optimal_route,dest)
    print()
    plt.show()
    return optimal_route


result_lst = GA(cities_name,cityList,it, population_size=10,  eliteSize=5, mutat_rate=0.01,generations=500)
        




   



