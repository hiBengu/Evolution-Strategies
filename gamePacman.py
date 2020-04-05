from kivy.app import App
from kivy.core.window import Window
from kivy.metrics import sp
from kivy.uix.widget import Widget
from kivy.clock import Clock
from kivy import properties as kp

from collections import defaultdict
import math

spriteSize = sp(30)
cols = (Window.width / spriteSize)
rows = (Window.height / spriteSize)
control = 0.01

wallCoords = [
[1,1],[1,2],[1,3]
]

foodCoords = [
[-50,-50],[0,2],[3,5],[0,5],[0,6], [7,8]
]

monsterCoords = [
[5,6],[3,2],[4,3]
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
    speed = kp.NumericProperty(2)

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
        sprite.coord = self.player

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
            print("sa")
            sprite.coord = [x * int(spriteSize) + 6 for x in coord]
            if not sprite.parent:
                print("as")
                self.root.add_widget(sprite)

    def placeMonsters(self, *args):
        for index, coord in enumerate(monsterCoords):
            sprite = monsterSprites[index]
            sprite.coord = [x * int(spriteSize) for x in coord]
            if not sprite.parent:
                self.root.add_widget(sprite)

    def on_keyboard_down(self, _, __, key , *___):
        if (key == 82):
            self.player[1] = self.player[1] + self.speed
        if (key == 81):
            self.player[1] = self.player[1] - self.speed
        if (key == 80):
            self.player[0] = self.player[0] - self.speed
        if (key == 79):
            self.player[0] = self.player[0] + self.speed

    def checkEat(self, *args):
        self.foodDistanceList = []

        for index, food in enumerate(foodCoords):
            if index == 0:  # Dummmy food
                continue

            distX = (food[0] * self.spriteSize + 15) - (self.player[0] + 15)  # multiply with sprite size to get coordinate
            distY = (food[1] * self.spriteSize + 15) - (self.player[1] + 15)  # add 15 to find the center of food
            distance = math.sqrt(distX * distX + distY * distY) # abs distance

            if distance <= 24:
                self.root.remove_widget(foodSprites[list(foodSprites)[index]]) # This allow us to find the object with index value, not key
                foodCoords.pop(index)
                for key, value in dict(foodSprites).items():
                    if value == foodSprites[list(foodSprites)[index]]:
                        del foodSprites[key]
                        break

                print(foodSprites)
                continue

            self.foodDistanceList.append(distance)


    def checkDie(self, *args):
        self.monsterDistanceList = []

        for monster in monsterCoords:
            distX = (monster[0] * self.spriteSize + 15) - (self.player[0] + 15)  # multiply with sprite size to get coordinate
            distY = (monster[1] * self.spriteSize + 15) - (self.player[1] + 15)  # add 15 to find the center of food
            distance = math.sqrt(distX * distX + distY * distY) # abs distance

            if distance <= 30:
                print("öldün")

            self.monsterDistanceList.append(distance)


if __name__ == '__main__':
    Pacman().run()
