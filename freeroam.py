import os
import pygame as pg
from pygame.math import Vector2
import math
import random as rd
from collections import deque
import tensorflow.keras as keras
from PIL import Image
import cnh_freeroam as cs
import numpy as np
import pickle

WIN_WIDTH = 1600
WIN_HEIGHT = 900
WIN_DIMS = (WIN_WIDTH, WIN_HEIGHT)
WHITE = (255, 255, 255)

DISP = pg.display.set_mode(WIN_DIMS)
pg.display.set_caption("Self Driving Car")

img = Image.open(os.path.join("car.png"))
img.thumbnail((80, 80), Image.ANTIALIAS)
CAR_PLAYER = pg.image.fromstring(img.tobytes(), img.size, img.mode)

img = Image.open(os.path.join('car2.png'))
img.thumbnail((80, 80), Image.ANTIALIAS)
CAR_TRAFFIC = pg.image.fromstring(img.tobytes(), img.size, img.mode)

'''
ACTION SPACE:
0 - STAY
1 - ACCEL-LEFT
2 - ACCEL-RIGHT
TOTAL: 9
'''

def map_range(val, c_min, c_max, t_min, t_max):
    val_ratio = (val - c_min) / (c_max - c_min)

    new_val = val_ratio * (t_max - t_min) + t_min

    return new_val

class Line(object):
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def update(self, dx, dy):
        self.x1 += dx
        self.x2 += dx
        self.y1 += dy
        self.y2 += dy

g_lines = [
    Line(0, 0, WIN_WIDTH, 0),
    Line(WIN_WIDTH, 0, WIN_WIDTH, WIN_HEIGHT),
    Line(0, WIN_HEIGHT, WIN_WIDTH, WIN_HEIGHT),
    Line(0, 0, 0, WIN_HEIGHT)
]
'''
reward_rects = [
    pg.Rect(400, 225, 800, 200),
    pg.Rect(1000, 225, 200, 450),
    pg.Rect(400, 225, 200, 450),
    pg.Rect(400, 475, 800, 200)
]
'''

class TrafficCar(object):
    def __init__(self, img, x, y, dx, dy, orientation):
        self.img = img;
        self.mask = pg.mask.from_surface(img)
        self.rect = self.img.get_rect()
        self.rect.center = (self.x, self.y) = (x, y)
        self.dx = dx
        self.dy = dy
        self.orientation = orientation

        x_off = 40
        y_off = 20

        if orientation == 'top' or orientation == 'bottom':
            x_off, y_off = y_off, x_off
            rr1 = pg.Rect(x, y, 40, 80)
            rr1.center = (x + 80, y)
            rr2 = pg.Rect(x, y, 40, 80)
            rr2.center = (x - 80, y)
            self.reward_rects = [rr1, rr2]

        else:
            rr1 = pg.Rect(x, y, 80, 40)
            rr1.center = (x, y + 80)
            rr2 = pg.Rect(x, y, 80, 40)
            rr2.center = (x, y - 80)
            self.reward_rects = [rr1, rr2]
        

        self.lines = [
            Line(x - x_off, y - y_off, x + x_off, y - y_off),
            Line(x + x_off, y - y_off, x + x_off, y + y_off),
            Line(x - x_off, y + y_off, x + x_off, y + y_off),
            Line(x - x_off, y - y_off, x - x_off, y + y_off)
        ]

        for line in self.lines:
            g_lines.append(line)

    def is_colliding(self, player):
        y_offset_sq = (self.rect.centery - player.rect.centery) ** 2
        x_offset_sq = (self.rect.centerx - player.rect.centerx) ** 2

        return (x_offset_sq + y_offset_sq) ** 0.5 < 60

    def update_position(self):
        self.x += self.dx
        self.y += self.dy
        self.rect.center = (self.x, self.y)
        for line in self.lines:
            line.update(self.dx, self.dy)

        for rect in self.reward_rects:
            rect.x += self.dx
            rect.y += self.dy

    def destroy_lines(self):
        for line in self.lines:
            g_lines.remove(line)

    def render(self, win):
        win.blit(self.img, self.rect.topleft)

class Traffic(object):
    def __init__(self, density):
        self.cars = []
        self.density = density

        self.bot_img = CAR_TRAFFIC;
        self.right_img = pg.transform.rotate(CAR_TRAFFIC, 90)
        self.top_img = pg.transform.rotate(CAR_TRAFFIC, 180)
        self.left_img = pg.transform.rotate(CAR_TRAFFIC, 270)

    def create_car(self):
        if rd.random() < self.density:
            orientation = rd.choice(['top', 'left', 'bottom', 'right'])

            if orientation == 'top':
                car = TrafficCar(self.top_img, rd.randint(1, WIN_WIDTH), -40, 0, 5, 'top')

            elif orientation == 'right':
                car = TrafficCar(self.right_img, WIN_WIDTH + 40, rd.randint(1, WIN_HEIGHT), -5, 0, 'right')

            elif orientation == 'bottom':
                car = TrafficCar(self.bot_img, rd.randint(1, WIN_WIDTH), WIN_HEIGHT + 40, 0, -5, 'bottom')

            else:
                car = TrafficCar(self.left_img, -40, rd.randint(1, WIN_HEIGHT), 5, 0, 'left')

            self.cars.append(car)

    def filter_cars(self):
        for car in self.cars:
            cond = (car.orientation == 'top' and car.y >= WIN_HEIGHT) or \
                (car.orientation == 'right' and car.x <= 0) or \
                    (car.orientation == 'bottom' and car.y <= 0) or \
                        (car.orientation == 'left' and car.x >= WIN_WIDTH)

            if cond:
                car.destroy_lines()
                self.cars.remove(car)

    def reset(self):
        for car in self.cars:
            car.destroy_lines()
        self.cars = []

    def check_collision(self, player):
        for car in self.cars:
            if car.is_colliding(player):
                return True

        return False

    def update(self):
        self.create_car()
        self.filter_cars()
        for car in self.cars:
            car.update_position()

    def get_rewards(self, player):
        for car in self.cars:
            for rect in car.reward_rects:
                if player.rect.colliderect(rect):
                    return 5
        '''
        for rect in reward_rects:
            if player.rect.colliderect(rect) or rect.contains(player.rect):
                return 50
        '''
        return 0

    def render(self, win):
        for car in self.cars:
            car.render(win)
                
class Ray(object):
    def __init__(self, x, y, angle):
        self.pos = Vector2(x, y)
        self.dir = Vector2(math.cos(math.radians(angle)), math.sin(math.radians(angle)))

    def update(self, vel):
        self.pos.x += vel.x
        self.pos.y += vel.y
    
    def rotate(self, degs):
        self.dir.rotate_ip(degs)

    def get_intersection_point(self, line: Line):
        x1 = line.x1
        y1 = line.y1
        x2 = line.x2
        y2 = line.y2

        x3 = self.pos.x
        y3 = self.pos.y
        x4 = x3 + self.dir.x
        y4 = y3 + self.dir.y

        den = ((x1 - x2) * (y3 - y4)) - ((y1 - y2) * (x3 - x4))

        if den == 0: return { 'pt': None, 'dist': None }

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den

        u = ((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den

        if t >= 0 and t <= 1 and u > 0:
            pt = Vector2(0, 0)
            pt.x = x1 + t * (x2 - x1)
            pt.y = y1 + t * (y2 - y1)

            return { 'pt': pt, 'dist': u }

        return { 'pt': None, 'dist': None }

class Sensor(object):
    def __init__(self, x, y, quantity):
        self.pos = Vector2(x, y)
        self.dir = Vector2(1, 0)
        self.rays = []
        step = 360 / quantity
        covered_angle = 0

        while covered_angle < 360:
            self.rays.append(Ray(x, y, covered_angle))
            covered_angle += step

    def update(self, vel):
        self.pos.x += vel.x
        self.pos.y += vel.y
        for ray in self.rays:
            ray.update(vel)

    def rotate(self, degs):
        self.dir.rotate_ip(degs)
        for ray in self.rays:
            ray.rotate(degs)

    def get_distance_data(self):
        data = []

        for ray in self.rays:
            min_dist = 999999
            closest = None

            for line in g_lines:
                pt_info = ray.get_intersection_point(line)

                if pt_info['pt'] == None: continue

                if pt_info['dist'] < min_dist:
                    min_dist = pt_info['dist']
                    closest = pt_info

            if closest == None:
                data.append(0)
                continue

            dist = (closest['dist'] / 300) if closest['dist'] <= 300 else 0
            # if dist > 0:
            #     pg.draw.line(DISP, WHITE, self.pos, closest['pt'])

            data.append(dist)

        return data

class PlayerCar(object):
    def __init__(self, img, x, y):
        self.img = img
        self.rect = self.img.get_rect()
        self.pos = Vector2(x, y)
        self.rect.center = (self.pos.x, self.pos.y)
        self.dir = Vector2(1, 0)
        self.vel = Vector2(0, 0)
        self.accel = Vector2(0, 0)
        self.angle = 0
        self.status = 'stopped'
        self.mask = pg.mask.from_surface(self.img)
        self.sensor = Sensor(x, y, 25)

    def accelarate(self):
        self.vel = self.vel + Vector2(self.dir.x * 0.8, self.dir.y * 0.8)
        self.status = 'accel'
        if self.vel.magnitude() > 7:
            self.vel.scale_to_length(7)

    def decelerate(self):
        self.vel = self.vel - self.dir
        self.status = 'decel'
        if self.vel.magnitude() > 5:
            self.vel.scale_to_length(5)

    def turn(self, direction):
        if self.status == 'accel':
            rot = map_range(self.vel.magnitude(), 0, 7, 0, 5)
        elif self.status == 'decel':
            rot = -map_range(self.vel.magnitude(), 0, 5, 0, 5)
        else:
            rot = 0

        if direction == 'left':
            self.dir = self.dir.rotate(-rot)
            self.angle = (self.angle + rot) % 360
            self.sensor.rotate(-rot)
            self.img = pg.transform.rotate(CAR_PLAYER, self.angle)

        elif direction == 'right':
            self.dir = self.dir.rotate(rot)
            self.sensor.rotate(rot)
            self.angle = (self.angle - rot) % 360
            self.img = pg.transform.rotate(CAR_PLAYER, self.angle)

        self.rect = self.img.get_rect()
        self.rect.center = (self.pos.x, self.pos.y)
        self.mask = pg.mask.from_surface(self.img)

    def update(self):
        self.pos += self.vel
        self.sensor.update(self.vel)
        self.rect.center = (self.pos.x, self.pos.y)
        self.vel *= 0.9
        if self.vel.magnitude() <= 0.5:
            self.status = 'stopped'
            self.vel = Vector2(0, 0)

    def get_sensor_data(self):
        return self.sensor.get_distance_data()

    def check_out_of_bounds(self):
        return self.pos.x <= WIN_WIDTH + 40 and \
            self.pos.x >= -40 and \
                self.pos.y <= WIN_HEIGHT + 40 and \
                    self.pos.y >= -40

    def render(self, win):
        win.blit(self.img, self.rect.topleft)
        #pg.draw.circle(win, WHITE, self.rect.center, 30)


class Environment(object):
    n_actions = 3
    input_dims = 30

    def __init__(self):
        self.player = PlayerCar(CAR_PLAYER, 800, 450)
        self.traffic = Traffic(0.03)

    def step(self, action):
        # if action in [1, 3, 4]:
        #     self.player.accelarate()
        
        # if action in [1, 4, 5]:
        #     decel = True
        #     self.player.decelerate()

        if action == 1:
            self.player.turn('left')

        if action == 2:
            self.player.turn('right')

        
        self.player.accelarate()

        self.player.update()
        self.traffic.update()

        state = self.player.get_sensor_data()
        state.append((self.player.angle) / 360)
        vel_norm = (self.player.vel.magnitude()) / 7
        state.append(math.floor(vel_norm if self.player.status == 'accel' else (-1 * vel_norm)))
        state.append((self.player.pos.x) / WIN_WIDTH)
        state.append((self.player.pos.y) / WIN_HEIGHT)
        
        done = self.traffic.check_collision(self.player) or not self.player.check_out_of_bounds()

        reward = -1000 if done else self.traffic.get_rewards(self.player)

        return state, reward, done

    def reset(self):
        self.player.pos.x = 800
        self.player.pos.y = 450
        self.traffic.reset()
        state = [0 for _ in range(30)]
        return state

    def render(self):
        DISP.fill((10, 10, 10))
        self.player.render(DISP)
        self.traffic.render(DISP)
        pg.display.update()


class ReplayBuffer(object):
    def __init__(self, size):
        self.size = size
        self.mem_ctr = 0
        self.states = deque(maxlen=size)
        self.actions = deque(maxlen=size)
        self.rewards = deque(maxlen=size)
        self.dones = deque(maxlen=size)
        self.next_states = deque(maxlen=size)

    def append(self, state, action, reward, state_, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(state_)
        self.dones.append(done)
        self.mem_ctr += 1

    def sample_buffer(self, batch_size):
        if self.mem_ctr < batch_size: return None

        indices = np.random.randint(self.mem_ctr % self.size, size=batch_size)

        states = np.array([self.states[i] for i in indices])
        actions = np.array([self.actions[i] for i in indices])
        rewards = np.array([self.rewards[i] for i in indices])
        next_states = np.array([self.next_states[i] for i in indices])
        dones = np.array([(1 - self.dones[i]) for i in indices])

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.states)

    def __getitem__(self, index):
        if index >= len(self.states): return None

        state = self.states[index]
        action = self.actions[index]
        reward = self.rewards[index]
        state_ = self.next_states[index]
        done = self.dones[index]

        return state, action, reward, state_, done

def build_dqn(input_dims, output_dims, learning_rate, hidden1, hidden2):
    model = keras.models.Sequential([
        keras.layers.Dense(hidden1, input_shape=(input_dims,)),
        keras.layers.Activation('relu'),
        keras.layers.Dense(hidden2),
        keras.layers.Activation('relu'),
        keras.layers.Dense(output_dims)
    ])

    model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate), loss='mse')

    return model


class DDQNAgent(object):
    def __init__(self, env):
        self.env = env
        self.n_actions = env.n_actions
        self.input_dims = env.input_dims
        self.epsilon = cs.epsilon_start
        self.epsilon_dec = cs.epsilon_dec
        self.mem_size = cs.replay_buffer_size
        self.memory = ReplayBuffer(self.mem_size)
        self.batch_size = cs.batch_size
        self.gamma = cs.gamma
        self.q_eval = build_dqn(self.input_dims, self.n_actions, cs.learning_rate, cs.hidden1, cs.hidden2)
        self.q_target = keras.models.clone_model(self.q_eval)

    def epsilon_greedy_policy(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)

        else:
            Q_values = self.q_eval.predict(np.array(state)[np.newaxis])
            return np.argmax(Q_values)

    def save_model(self):
        self.q_eval.save(cs.model_f_name, overwrite=True)
        with open(cs.rb_f_name, 'wb') as rb_file:
            pickle.dump(self.memory, rb_file)

    def load_model(self):
        self.q_eval = keras.models.load_model(cs.model_f_name)
        self.q_target = keras.models.load_model(cs.model_f_name)

        with open(cs.rb_f_name, 'rb') as rb_file:
            self.memory = pickle.load(rb_file)

    def step(self, state):
        action = self.epsilon_greedy_policy(state)
        next_state, reward, done = self.env.step(action)
        self.memory.append(state, action, reward, next_state, done)
        return next_state, reward, done

    def learning_step(self):
        states, actions, rewards, next_states, dones = self.memory.sample_buffer(self.batch_size)

        next_Q_values = self.q_target.predict(next_states)
        eval_Q_values = self.q_eval.predict(next_states)
        max_actions = np.argmax(eval_Q_values, axis=1)
        target_Q_values = self.q_eval.predict(states)

        batch_indices = np.arange(self.batch_size, dtype=np.int32)

        target_Q_values[batch_indices, actions] = rewards + self.gamma * \
            next_Q_values[batch_indices, max_actions.astype(int)] * dones

        self.q_eval.fit(states, target_Q_values, verbose=0)

    def eval(self):
        self.load_model()
        to_render = False

        while True:
            ep_done = False
            obs = self.env.reset()

            while not ep_done:
                q_vals = self.q_eval.predict(np.array(obs)[np.newaxis])
                action = np.argmax(q_vals[0])

                obs, reward, done = self.env.step(action)

                for event in pg.event.get():
                    if event.type == pg.QUIT:
                        pg.quit()
                        quit()

                    if event.type == pg.KEYDOWN:
                        if event.key == pg.K_r:
                            to_render = not to_render

                if to_render:
                    self.env.render()

    def train(self):
        exps_stored = 0
        learn_target = 100
        to_render = False

        while True:
            ep_done = False
            obs = self.env.reset()
            while not ep_done:
                obs, reward, done = self.step(obs)
                exps_stored += 1
                ep_done = done

                for event in pg.event.get():
                    if event.type == pg.QUIT:
                        pg.quit()
                        quit()

                    if event.type == pg.KEYDOWN:
                        if event.key == pg.K_r:
                            to_render = not to_render

                if to_render:
                    self.env.render()

                if exps_stored > self.batch_size:
                    self.learning_step()

                if exps_stored % learn_target == 0:
                    self.q_target.set_weights(self.q_eval.get_weights())

                if exps_stored % 1000 == 0:
                    self.save_model()

                self.epsilon = max(self.epsilon * self.epsilon_dec, 0.01)


    def commence(self, mode):
        if mode == '1':
            self.train()
        elif mode == '2':
            self.load_model()
            self.epsilon = 0.01
            self.train()
        elif mode == '3':
            self.eval()
        

agent = DDQNAgent(Environment())

mode = input("1 = Train, 2 = Cont. Train, 3 = Eval: ")

agent.commence(mode)

def main_loop():

    player = PlayerCar(CAR_PLAYER, 200, 200)
    traffic = Traffic(0.03)
    clock = pg.time.Clock()

    while True:
        clock.tick(60)

        keys = pg.key.get_pressed()

        if keys[pg.K_LEFT]:
            player.turn('left')
        elif keys[pg.K_RIGHT]:
            player.turn('right')

        if keys[pg.K_UP]:
            player.accelarate()
        elif keys[pg.K_DOWN]:
            player.decelerate()

        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                quit()

        if traffic.check_collision(player):
            traffic.reset()

        traffic.update()
        player.update()

        draw_game(player, traffic)

#main_loop()
