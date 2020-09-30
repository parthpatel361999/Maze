import random as rnd
from queue import LifoQueue, PriorityQueue

import numpy as np

"""
Define a Cell class. This class keeps track of the previous Cell through which the search algorithm traveled to analyze this Cell.

"""
class Cell: 
    
    OPEN = 0
    BLOCKED = -1
    ON_FIRE = 2
    
    def __init__(self, coords, prev = None, f = 0, g = 0, h = -1):
        self.coords = coords
        self.prev = prev
        self.f = f
        self.g = g
        self.h = h
        
    def __lt__(self, other):   
        if self.f == other.f:
            if self.h == other.h:
                if self.g == other.g:
                    return self.coords > other.coords
                return self.g < other.g
            return 
        return self.f < other.f
    
    def __eq__(self, other):
        return self.coords == other.coords
    
    def __repr__(self):
        return str(self.coords) + " :: parent=" + ("NONE" if self.prev is None else str(self.prev.coords))

class Maze:
    def __init__(self, dim, p, q = 0):
        self.board =  np.zeros([dim, dim], dtype=int)
        self.dim = dim
        self.currentfire = []
        self.q = q
        self.p = p
        obstaclesList = []
        for i in range(0, dim): #range() generates a list from 0 to dim-1 which can take up a lot of memory
            for j in range(0, dim):             
                if(rnd.random() < p):
                    self.board[i, j] = Cell.BLOCKED
                    obstaclesList.append((i, j))
        self.obstacles = np.array(obstaclesList)
        self.board[0, 0] = Cell.OPEN
        self.board[dim - 1, dim - 1] = Cell.OPEN

    def isSolvable(self):
        return pathExists(self)
    
    def startFire(self):
        i = rnd.randint(0,self.dim-1)
        j = rnd.randint(0,self.dim-1)
        while(self.board[i, j] == -1 or (i == 0 and j == 0) or (i == self.dim - 1 and j == self.dim - 1)):
            i = rnd.randint(0, self.dim - 1)
            j = rnd.randint(0, self.dim - 1)
        self.board[i,j] = Cell.ON_FIRE
        self.currentfire.append([i, j, None])
        
    def resetFire(self):
        if (len(self.currentfire) == 1):
            fireRow, fireCol, ignored = self.currentfire[0]
            self.board[fireRow, fireCol] = Cell.OPEN
            self.currentfire = []  
        
    def isFireReachableToAgent(self):
        if (len(self.currentfire) == 1): 
            fireRow, fireCol, ignored = self.currentfire[0]
            return pathExists(self,fromCoords=(0, 0), toCoords=(fireRow, fireCol))
    
    def iterateFire(self):
        visited = [] #neighbors already visited in current iteration
        nowonfire = [] #need to append all new fires at the end 
        for f in self.currentfire:
            if(f[2] != 0): #check to see if all neighbors are already on fire. 
                neighbors = findNeighboringFlammableCoords(f[0:2],self) #calculate valid neighbors of f, the cell that is currently on fire
                ofnbor = 0 #counter to find out how many neighbors are on fire
                for n in neighbors: #loop through the neighbors of f
                    nr,nc = n 
                    if (self.board[nr,nc] == Cell.ON_FIRE): 
                        ofnbor = ofnbor + 1 
                    else:
                        if(n[0:2] not in visited): #only roll once for each cell not on fire
                            if(self.board[nr,nc] != Cell.BLOCKED):
                                surroundings = findNeighboringFlammableCoords(n[0:2],self) #see what the neighbors are for the neighbor of f
                                k = 0 #k will be guaranteed to be at least 1
                                for nn in surroundings:
                                    nnr,nnc = nn
                                    if(self.board[nnr,nnc] == Cell.ON_FIRE):
                                        k = k + 1 #counter
                                if(rnd.random() <= (1-((1-self.q)**k))): #roll
                                    self.board[nr,nc] = Cell.ON_FIRE 
                                    nowonfire.append([nr,nc,None])
                                    ofnbor = ofnbor + 1
                                else:
                                    visited.append([nr,nc])
                if(ofnbor == len(neighbors)):
                        f[2] = 0
        if(nowonfire):
            for x in nowonfire:
                self.currentfire.append(x)
            return True
        return False

def pathExists(maze, fromCoords = (0,0), toCoords = None):
    if toCoords is None:
        toCoords = (maze.dim - 1, maze.dim - 1)
    toRow, toCol = toCoords
    visited = set()
    fringe = LifoQueue()
    fringe.put(fromCoords)
    while not fringe.empty():
        currentCoords = fringe.get()
        currentRow, currentCol = currentCoords
        if (currentRow == toRow and currentCol == toCol):
            return True
        else:
            neighbors = findNeighboringOpenCoords(currentCoords, maze)
            for neighbor in neighbors:
                if neighbor in visited:
                    continue
                fringe.put(neighbor)
            visited.add(currentCoords)
    return False

def findNeighboringOpenCoords(coords, maze):
    cellRow, cellCol = coords
    potentialNeighbors = [(cellRow + 1, cellCol), (cellRow - 1, cellCol), (cellRow, cellCol - 1), (cellRow, cellCol + 1)]
    neighbors = []
    for potentialNeighbor in potentialNeighbors:
        row, col = potentialNeighbor
        if (row >= maze.dim or row < 0 or col >= maze.dim or col < 0 or maze.board[row,col] != Cell.OPEN):
            continue
        neighbors.append(potentialNeighbor)
    return neighbors

def findNeighboringFlammableCoords(coords,maze): 
    cellRow, cellCol = coords
    potentialNeighbors = [(cellRow + 1, cellCol), (cellRow - 1, cellCol), (cellRow, cellCol - 1), (cellRow, cellCol + 1)]
    neighbors = []
    for potentialNeighbor in potentialNeighbors:
        row, col = potentialNeighbor
        if (row >= maze.dim or row < 0 or col >= maze.dim or col < 0 or maze.board[row,col] == Cell.BLOCKED):
            continue
        neighbors.append(potentialNeighbor)
    return neighbors

def uniformCost(cell, maze, visited):
    return 0

def shortestPathSearch(maze, startCoords = (0, 0), goalCoords = None, heuristicFunction = uniformCost, findNeighborsFunction = findNeighboringOpenCoords):
    if goalCoords is None:
        goalCoords = (maze.dim - 1, maze.dim - 1)
    startCell = Cell(startCoords)
    visited = set()
    fringe = PriorityQueue()
    fringe.put(startCell)
    heuristicVisited = {}
    expandedCells = 0
    while not fringe.empty():
        currentCell = fringe.get()
        if (currentCell.coords) in visited:
            continue
        if (currentCell.coords == goalCoords):
            shortestPath = []
            while (currentCell is not None):
                shortestPath.append(currentCell.coords)
                currentCell = currentCell.prev
            shortestPath.reverse()
            return (shortestPath, expandedCells)
        else:
            neighborsCoordsList = findNeighborsFunction(coords = currentCell.coords, maze = maze)
            for neighborCoords in neighborsCoordsList:
                if neighborCoords in visited:
                    continue
                neighbor = Cell(neighborCoords, prev = currentCell)
                neighbor.g = currentCell.g + 1
                neighbor.h = heuristicFunction(cell = neighbor, maze = maze, visited = heuristicVisited)
                neighbor.f = neighbor.g + neighbor.h
                fringe.put(neighbor)
            visited.add(currentCell.coords)
            expandedCells += 1
    return None
