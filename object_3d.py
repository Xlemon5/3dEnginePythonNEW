import pygame as pg
from matrix_functions import *
from numba import njit
import numpy as np

@njit(fastmath=True)
def any_func(arr, a, b):
    return np.any((arr == a) | (arr == b))

class Object3d():
    def __init__(self, render) -> None:
        self.render = render
        self.vertices = np.array([
            (0, 0, 0, 1), (0, 1, 0, 1), (1, 1, 0, 1), (1, 0, 0, 1),
            (0, 0, 1, 1), (0, 1, 1, 1), (1, 1, 1, 1), (1, 0, 1, 1)])
        self.faces = np.array([(0, 1, 2, 3), (4,5,6,7), (0,4,5,1), (2,3,7,6), (1,2,6,5), (0,3,7,4)])
        
        self.font = pg.font.SysFont('Arial', 30, bold=True)
        self.color_faces = [(pg.Color('orange'), face) for face in self.faces]
        self.movement_flag, self.draw_vertices = True, False
        self.label = ''
        
        # Новые атрибуты для управления
        self.selected_axis = 'y'
        self.selected_vertex = 0
        self.scale_factor = 1.0
        self.projection_plane = 'xy'  # Плоскость проекции
        self.rotation_mode = 'axis'  # 'axis' или 'vertex'
        self.scale_mode = False  # Режим масштабирования
        
        # Атрибут для сохранения состояния объекта
        self.saved_vertices = None

        # Дополнительные атрибуты
        self.plane_changed = False
        self.projection_applied = False
        self.vertex_rotation_axis = 'y'

    def save_state(self):
        """Сохраняет текущее состояние вершин объекта для последующего сброса."""
        self.saved_vertices = self.vertices.copy()

    def control(self):
        key = pg.key.get_pressed()
        # Выбор оси вращения
        if key[pg.K_x]:
            self.selected_axis = 'x'
        if key[pg.K_y]:
            self.selected_axis = 'y'
        if key[pg.K_z]:
            self.selected_axis = 'z'

        # Вращение вокруг выбранной оси
        if self.rotation_mode == 'axis':
            if key[pg.K_u]:
                self.rotate_around_axis(-0.05)
            if key[pg.K_i]:
                self.rotate_around_axis(0.05)

        # Переключение режима вращения вокруг вершины
        if key[pg.K_v]:
            self.rotation_mode = 'vertex' if self.rotation_mode == 'axis' else 'axis'

        # Вращение вокруг выбранной вершины
        if self.rotation_mode == 'vertex':
            if key[pg.K_n]:
                self.rotate_around_vertex(self.selected_vertex, -0.05)
            if key[pg.K_m]:
                self.rotate_around_vertex(self.selected_vertex, 0.05)
            # Выбор оси вращения вокруг вершины
            if key[pg.K_8]:
                self.vertex_rotation_axis = 'x'
            if key[pg.K_9]:
                self.vertex_rotation_axis = 'y'
            if key[pg.K_0]:
                self.vertex_rotation_axis = 'z'
            # Выбор вершины
            if key[pg.K_4]:
                self.selected_vertex = 4
            if key[pg.K_5]:
                self.selected_vertex = 5
            if key[pg.K_6]:
                self.selected_vertex = 6
            if key[pg.K_7]:
                self.selected_vertex = 7

        # Переключение режима масштабирования
        if key[pg.K_l]:
            self.scale_mode = not self.scale_mode

        # Масштабирование относительно выбранной плоскости
        if self.scale_mode:
            if key[pg.K_LEFTBRACKET]:  # Клавиша '['
                self.scale_relative_to_plane(self.projection_plane, 0.99)
            if key[pg.K_RIGHTBRACKET]:  # Клавиша ']'
                self.scale_relative_to_plane(self.projection_plane, 1.01)

        # Выбор плоскости проекции
        if not self.plane_changed:
            if key[pg.K_1]:
                self.projection_plane = 'xy'
                self.plane_changed = True
            elif key[pg.K_2]:
                self.projection_plane = 'xz'
                self.plane_changed = True
            elif key[pg.K_3]:
                self.projection_plane = 'yz'
                self.plane_changed = True
        if not key[pg.K_1] and not key[pg.K_2] and not key[pg.K_3]:
            self.plane_changed = False

        # Проекция на плоскость
        if key[pg.K_p] and not self.projection_applied:
            self.project_onto_plane(self.projection_plane)
            self.projection_applied = True
        if not key[pg.K_p]:
            self.projection_applied = False

        # Сброс объекта
        if key[pg.K_r]:
            self.reset()

    def draw(self):
        self.screen_projection()
        self.movement()

    def movement(self):
        if self.movement_flag:
            pass  # Убираем автоматическое вращение
            
    def screen_projection(self):
        vertices = self.vertices @ self.render.camera.camera_matrix()
        vertices = vertices @ self.render.projection.projection_matrix
        vertices /= vertices[:, -1].reshape(-1, 1)
        vertices[(vertices > 2) | (vertices < -2)] = 0
        vertices = vertices @ self.render.projection.to_screen_matrix
        vertices = vertices[:, :2]

        for index, color_face in enumerate(self.color_faces):
            color, face = color_face
            polygon = vertices[face]
            if not any_func(polygon, self.render.h_width, self.render.h_height):
                pg.draw.polygon(self.render.screen, color, polygon, 1)
                if self.label:
                    text = self.font.render(self.label[index], True, pg.Color('white'))
                    self.render.screen.blit(text, polygon[-1])

        if self.draw_vertices:
            for vertex in vertices:
                if not any_func(vertex, self.render.h_width, self.render.h_height):
                    pg.draw.circle(self.render.screen, pg.Color('white'), vertex, 2)
        
    def translate(self, pos):
        self.vertices = self.vertices @ translate(pos)

    def scale(self, scale_to):
        self.vertices = self.vertices @ scale(scale_to)

    def rotate_x(self, angle):
        self.vertices = self.vertices @ rotate_x(angle)

    def rotate_y(self, angle):
        self.vertices = self.vertices @ rotate_y(angle)

    def rotate_z(self, angle):
        self.vertices = self.vertices @ rotate_z(angle)

    def reset(self):
        """Сбрасывает объект к сохраненному состоянию после начальных преобразований."""
        if self.saved_vertices is not None:
            self.vertices = self.saved_vertices.copy()
            # Сбрасываем дополнительные параметры
            self.selected_axis = 'y'
            self.selected_vertex = 0
            self.scale_factor = 1.0
            self.projection_plane = 'xy'
            self.rotation_mode = 'axis'
            self.scale_mode = False
            self.vertex_rotation_axis = 'y'

    def rotate_around_axis(self, angle):
        if self.selected_axis == 'x':
            self.rotate_x(angle)
        elif self.selected_axis == 'y':
            self.rotate_y(angle)
        elif self.selected_axis == 'z':
            self.rotate_z(angle)

    def rotate_around_vertex(self, vertex_index, angle):
        vertex = self.vertices[vertex_index][:3]
        # Перенос объекта так, чтобы выбранная вершина была в начале координат
        self.translate(-vertex)
        # Выполняем вращение вокруг выбранной оси
        if self.vertex_rotation_axis == 'x':
            self.rotate_x(angle)
        elif self.vertex_rotation_axis == 'y':
            self.rotate_y(angle)
        elif self.vertex_rotation_axis == 'z':
            self.rotate_z(angle)
        # Возврат объекта на место
        self.translate(vertex)

    def project_onto_plane(self, plane):
        for i in range(len(self.vertices)):
            x, y, z, w = self.vertices[i]
            if plane == 'xy':
                self.vertices[i] = (x, y, 0, w)
            elif plane == 'xz':
                self.vertices[i] = (x, 0, z, w)
            elif plane == 'yz':
                self.vertices[i] = (0, y, z, w)

    def scale_relative_to_plane(self, plane, scale_factor):
        # Масштабирование вдоль осей, не лежащих в плоскости
        if plane == 'xy':
            scale_matrix = np.array([
                [scale_factor, 0, 0, 0],
                [0, scale_factor, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
        elif plane == 'xz':
            scale_matrix = np.array([
                [scale_factor, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, scale_factor, 0],
                [0, 0, 0, 1]
            ])
        elif plane == 'yz':
            scale_matrix = np.array([
                [1, 0, 0, 0],
                [0, scale_factor, 0, 0],
                [0, 0, scale_factor, 0],
                [0, 0, 0, 1]
            ])
        self.vertices = self.vertices @ scale_matrix

class Axes(Object3d):
    def __init__(self, render):
        super().__init__(render)
        self.vertices = np.array([(0, 0, 0, 1), (1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)])
        self.faces = np.array([(0, 1), (0, 2), (0, 3)])
        self.colors = [pg.Color('red'), pg.Color('green'), pg.Color('blue')]
        self.color_faces = [(color, face) for color, face in zip(self.colors, self.faces)]
        self.draw_vertices = False
        self.label = 'XYZ'
