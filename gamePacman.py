from kivy.app import App
from kivy.core.window import Window
from kivy.metrics import sp
from kivy.uix.widget import Widget
from kivy.clock import Clock
from kivy import properties as kp

from collections import defaultdict


spriteSize = sp(30)
cols = (Window.width / spriteSize)
rows = (Window.height / spriteSize)

wallCoords = [
[1,1],[1,2],[1,3]
]

foodCoords = [
[0,2],[0,5],[0,6]
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
    speed = kp.NumericProperty(5)

    player = kp.ListProperty([0,0])

    def on_start(self):
        Window.bind(on_keyboard=self.on_keyboard_down)

        self.on_player()
        self.buildWalls()
        self.placeFoods()
        self.placeMonsters()

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
            sprite.coord = [x * int(spriteSize) + 6 for x in coord]
            if not sprite.parent:
                self.root.add_widget(sprite)

    def placeMonsters(self, *args):
        for index, coord in enumerate(monsterCoords):
            sprite = monsterSprites[index]
            sprite.coord = [x * int(spriteSize) + 6 for x in coord]
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


if __name__ == '__main__':
    Pacman().run()
