import pygame as pg
import math
from object_3d import *
from camera import *
from projection import *


class SoftwareRender():
    def __init__(self) -> None:
        pg.init()
        self.res = self.width, self.height = 1600,  900
        self.h_width, self.h_height = self.width // 2, self.height // 2
        self.fps = 60
        self.screen = pg.display.set_mode(self.res)
        self.clock = pg.time.Clock()
        self.create_objects()
        
    def create_objects(self):
        self.camera = Camera(self, [0.5, 1, -4])
        self.projection = Projection(self)
        self.object = Object3d(self)
        self.object.translate([1, 0.4, 0.2])
        self.object.rotate_y(-math.pi / 6)
        self.object.save_state()  # Сохраняем состояние объекта после начальных преобразований
        self.axes = Axes(self)
        self.axes.translate([0.7, 0.9, 0.7])
        self.world_axes = Axes(self)
        self.world_axes.movement_flag = False
        self.world_axes.scale(2.5)
        self.world_axes.translate([0.0001, 0.0001, 0.0001])
        
    def draw(self):
        self.screen.fill(pg.Color('darkslategray'))
        self.world_axes.draw()
        self.axes.draw()
        self.object.draw()
    
    def run(self):
        while True:
            self.draw()
            self.camera.control()
            self.object.control()
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    exit()
            pg.display.set_caption(str(self.clock.get_fps()))
            pg.display.flip()
            self.clock.tick(self.fps)
            
if __name__ == "__main__":
    app = SoftwareRender()
    app.run()
