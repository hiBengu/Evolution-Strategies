from kivy.app import App
from kivy.core.window import Window
from kivy.metrics import sp
from kivy.uix.widget import Widget
from kivy.clock import Clock
from kivy import properties as kp

from collections import defaultdict
import math

Window.size = (600, 400)

spriteSize = sp(30)
cols = (Window.width / spriteSize)
rows = (Window.height / spriteSize)
control = 0.01

wallCoords = [
[1,1],[2,1],[3,1],[4,1],[6,1],[8,1],[9,1],[10,1],[11,1],[12,1],[13,1],[14,1],[16,1],[17,1],[18,1],
[6,2],[8,2],[14,2],
[1,3],[2,3],[3,3],[4,3],[6,3],[8,3],[10,3],[11,3],[12,3],[14,3],[15,3],[17,3],[18,3],[19,3],
[10,4],[0,5],
[1,5],[2,5],[3,5],[4,5],[6,5],[8,5],[9,5],[10,5],[11,5],[12,5],[13,5],[14,5],[16,5],[17,5],[18,5],[5,5],
[1,7],[2,7],[4,7],[5,7],[6,7],[8,7],[9,7],[10,7],[11,7],[12,7],[14,7],[15,7],[16,7],[17,7],[18,7],[7,7],
[11,8],[17,8],
[1,9],[3,9],[4,9],[5,9],[6,9],[0,9],[9,9],[10,9],[11,9],[12,9],[14,9],[13,9],[16,9],[17,9],[19,9],[7,9],
[5,10],[12,10],
[1,11],[2,11],[3,11],[4,11],[6,11],[8,11],[9,11],[10,11],[11,11],[12,11],[13,11],[14,11],[16,11],[17,11],[18,11],[5,11],
]

foodCoords = [[-1,-1]]

for x in range(20):
    for y in range(13):
        if([x,y] not in wallCoords and [x,y] != [0,0] and [x,y] != [10,2]):
            foodCoords.append([x,y])

monsterCoords = [
[10,2], [10,2]
]

class wallRect(Widget):
    coord = kp.ListProperty([0,0])
    bgcolor = kp.ListProperty([0,0,0,0])

wallSprites = defaultdict(lambda: wallRect())

class foodRect(Widget):
    coord = kp.ListProperty([0,0])
    bgcolor = kp.ListProperty([0,0,0,0])

foodSprites = defaultdict(lambda: foodRect())

class playerRect(Widget):
    coord = kp.ListProperty([0,0])
    bgcolor = kp.ListProperty([0,0,0,0])

playerSprites = defaultdict(lambda: playerRect())

class monsterRect(Widget):
    coord = kp.ListProperty([0,0])
    bgcolor = kp.ListProperty([0,0,0,0])

monsterSprites = defaultdict(lambda: monsterRect())

class Pacman(App):
    spriteSize = kp.NumericProperty(spriteSize)

    player = kp.ListProperty([0,0])

    def on_start(self):
        Window.bind(on_keyboard=self.on_keyboard_down)

        self.on_player()
        self.buildWalls()
        self.placeFoods()
        self.placeMonsters()
        Clock.schedule_interval(self.checkEat, control)
        Clock.schedule_interval(self.checkDie, control)

    def on_player(self, *args):
        sprite = playerSprites[0]
        sprite.coord = [x * int(spriteSize) + 6 for x in self.player]

        if not sprite.parent:
            self.root.add_widget(sprite)

    def buildWalls(self, *args):
        for index, coord in enumerate(wallCoords):
            sprite = wallSprites[index]
            sprite.coord = [x * int(spriteSize) for x in coord]
            if not sprite.parent:
                self.root.add_widget(sprite)

    def placeFoods(self, *args):
        for index, coord in enumerate(foodCoords):
            sprite = foodSprites[index]
            print(sprite)
            sprite.coord = [x * int(spriteSize) + 12 for x in coord]
            if not sprite.parent:
                self.root.add_widget(sprite)

    def placeMonsters(self, *args):
        for index, coord in enumerate(monsterCoords):
            sprite = monsterSprites[index]
            sprite.coord = [x * int(spriteSize) + 6for x in coord]
            if not sprite.parent:
                self.root.add_widget(sprite)

    def on_keyboard_down(self, _, __, key , *___):
        if (key == 82):
            if (self.checkWall([0,1])):
                self.player[1] = self.player[1] + 1
        if (key == 81):
            if (self.checkWall([0,-1])):
                self.player[1] = self.player[1] - 1
        if (key == 80):
            if (self.checkWall([-1,0])):
                self.player[0] = self.player[0] - 1
        if (key == 79):
            if (self.checkWall([1,0])):
                self.player[0] = self.player[0] + 1

    def checkEat(self, *args):
        self.foodDistanceList = []

        for index, food in enumerate(foodCoords):
            if index == 0:  # Dummmy food
                continue

            distX = (food[0] * self.spriteSize) - (self.player[0] * self.spriteSize)  # multiply with sprite size to get coordinate
            distY = (food[1] * self.spriteSize) - (self.player[1] * self.spriteSize)  # add 15 to find the center of food
            distance = math.sqrt(distX * distX + distY * distY) # abs distance

            if distance <= 1:
                self.root.remove_widget(foodSprites[list(foodSprites)[index]]) # This allow us to find the object with index value, not key
                foodCoords.pop(index)
                for key, value in dict(foodSprites).items():
                    if value == foodSprites[list(foodSprites)[index]]:
                        del foodSprites[key]
                        break

                continue

            self.foodDistanceList.append(distance)


    def checkDie(self, *args):
        self.monsterDistanceList = []

        for monster in monsterCoords:
            distX = (monster[0] * self.spriteSize) - (self.player[0] * self.spriteSize)  # multiply with sprite size to get coordinate
            distY = (monster[1] * self.spriteSize) - (self.player[1] * self.spriteSize)  # add 15 to find the center of food
            distance = math.sqrt(distX * distX + distY * distY) # abs distance
            if distance <= 1:
                print("öldün")

            self.monsterDistanceList.append(distance)

    def checkWall(self, nextMove):
        for wall in wallCoords:
            distX = (wall[0] * self.spriteSize) - ((self.player[0]+nextMove[0])*self.spriteSize)  # multiply with sprite size to get coordinate
            distY = (wall[1] * self.spriteSize) - ((self.player[1]+nextMove[1])*self.spriteSize)  # add 15 to find the center of wall
            distance = math.sqrt(distX * distX + distY * distY) # abs distance

            mapBoundCheckNeg = (self.player[0]+nextMove[0] < 0) or (self.player[1]+nextMove[1] < 0)
            mapBoundCheckPos = (self.player[0]+nextMove[0]+1 > Window.width / self.spriteSize) or ((self.player[1]+nextMove[1]+1 > Window.height / self.spriteSize))
            if (distance <= 1 or mapBoundCheckNeg or mapBoundCheckPos):
                print("gidemezsin")
                return False
        return True

if __name__ == '__main__':
    Pacman().run()
