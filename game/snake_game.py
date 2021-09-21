import random
import numpy as np
import game.constants as constants
from game.helpers import Point, Direction

import pygame

try:
    from ai.constants import STEP_REWARD, GAME_OVER_PENALTY, FOOD_REWARD
except ImportError:
    FOOD_REWARD = 100
    GAME_OVER_PENALTY = 100
    STEP_REWARD = 1


class SnakeGame:
    def __init__(self):
        successes, failures = pygame.init()
        print("{0} successes and {1} failures".format(successes, failures))
        pygame.font.init()
        pygame.display.set_caption('Snake game')

        self.font = pygame.font.SysFont('Comic Sans MS', constants.FONT_SIZE)
        self.screen = pygame.display.set_mode((constants.WIDTH, constants.HEIGHT))
        self.clock = pygame.time.Clock()
        self.screen.fill(constants.BLACK)

        self.direction = None
        self.step_taken = None
        self.snake = None
        self.head = None
        self.food = None
        self.game_over = None
        self.prev_distance = None
        self.score = None
        self.reward = None
        self.food_count = None

        self.reset()

    def reset(self):
        self.score = 0
        self.reward = 0
        self.direction = Direction.RIGHT
        self.step_taken = 0
        self.head = Point(constants.BLOCK_SIZE * 4, constants.HEIGHT / 2)
        self.snake = [self.head, Point(constants.BLOCK_SIZE * 3, constants.HEIGHT // 2)]

        self.food_count = 0
        self.__generate_food()
        self.prev_distance = self.__shortest_dist()

        self.game_over = False

    def play(self):
        while not self.game_over:
            self.__play_step()

    def play_ai(self, ai_action, ):
        self.__play_step(ai_action)
        return self.reward, self.game_over, self.score

    def get_state(self):
        danger_straight = self.__is_collision(self.__move_point(self.head), snake_check=True)
        danger_left = self.__is_collision(self.__move_point(self.head, direction=self.__anti_clockwise_turn()),
                                          snake_check=True)
        danger_right = self.__is_collision(self.__move_point(self.head, direction=self.__clockwise_turn()),
                                           snake_check=True)

        food_up = False
        food_down = False
        food_left = False
        food_right = False

        if self.food.x > self.head.x:
            food_right = True
        if self.food.x < self.head.x:
            food_left = True
        if self.food.y > self.head.y:
            food_down = True
        if self.food.y < self.head.y:
            food_up = True

        direction_up = self.direction == Direction.UP
        direction_down = self.direction == Direction.DOWN
        direction_left = self.direction == Direction.LEFT
        direction_right = self.direction == Direction.RIGHT

        state = np.array([danger_straight, danger_left, danger_right,
                          direction_up, direction_left, direction_down, direction_right,
                          food_up, food_left, food_down, food_right])
        return state

    def __play_step(self, ai_action=None):
        self.reward = 0
        self.clock.tick(constants.FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.__death()
                quit()
            elif event.type == pygame.KEYDOWN and ai_action is None:
                self.__change_key_direction(event.key)
                break
        if ai_action is not None:
            self.__change_ai_direction(ai_action)
        if self.__is_collision() or self.step_taken >= 100 * len(self.snake):
            self.reward -= GAME_OVER_PENALTY
            self.__death()
            return
        elif len(self.snake) >= constants.WIDTH * constants.HEIGHT:
            self.score = 1e10
            self.reward += 1e10
            self.__death()
            return
        self.__move()
        self.__update_ui()

    def __death(self):
        print(f'Score: {self.score}, Food: {self.food_count}')
        self.game_over = True

    def __generate_food(self):
        x = random.randint(1, (constants.WIDTH // constants.BLOCK_SIZE) - 1) * constants.BLOCK_SIZE
        y = random.randint(1, (constants.HEIGHT // constants.BLOCK_SIZE) - 1) * constants.BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self.__generate_food()
        self.food_count += 1

    def __change_key_direction(self, key):
        if key == pygame.K_w and self.direction != Direction.DOWN:
            self.direction = Direction.UP
        elif key == pygame.K_s and self.direction != Direction.UP:
            self.direction = Direction.DOWN
        elif key == pygame.K_a and self.direction != Direction.RIGHT:
            self.direction = Direction.LEFT
        elif key == pygame.K_d and self.direction != Direction.LEFT:
            self.direction = Direction.RIGHT

    def __is_collision(self, pt: Point = None, snake_check=not constants.SNAKE_EAT_ITSELF):
        if pt is None:
            pt = self.head
        if pt.x >= constants.WIDTH or pt.x < 0:
            return True
        elif pt.y >= constants.HEIGHT or pt.y < 0:
            return True
        elif snake_check and pt in self.snake[1:]:
            return True
        return False

    def __move(self):
        self.head = self.__move_point(self.head)
        self.snake.insert(0, self.head)
        if self.head == self.food:  # ate food
            self.reward += FOOD_REWARD
            self.score += 1
            self.step_taken = 0
            self.__generate_food()
        else:
            self.snake.pop()
            if self.head in self.snake[1:] and constants.SNAKE_EAT_ITSELF:  # ate itself
                idx = self.snake.index(self.head, 1)
                cut = abs(idx - len(self.snake))
                self.reward -= cut * FOOD_REWARD
                self.score -= cut
                self.snake = self.snake[:idx]

            if self.__shortest_dist() < self.prev_distance:
                self.reward += STEP_REWARD
            else:
                self.reward -= STEP_REWARD
            self.prev_distance = self.__shortest_dist()
        self.step_taken += 1

    def __move_point(self, pt: Point, direction: Direction = None):
        if direction is None:
            direction = self.direction
        x = pt.x
        y = pt.y
        if direction == Direction.UP:
            y -= constants.BLOCK_SIZE
        elif direction == Direction.DOWN:
            y += constants.BLOCK_SIZE
        if direction == Direction.RIGHT:
            x += constants.BLOCK_SIZE
        if direction == Direction.LEFT:
            x -= constants.BLOCK_SIZE
        return Point(x, y)

    def __update_ui(self):
        self.screen.fill(constants.BLACK)
        for pt in self.snake:
            pygame.draw.rect(self.screen, constants.BLUE, pygame.Rect(pt.x, pt.y, constants.BLOCK_SIZE,
                                                                      constants.BLOCK_SIZE))
            if pt == self.head:
                eye_size = constants.BLOCK_SIZE // 2
                pygame.draw.rect(self.screen, constants.RED,
                                 pygame.Rect(pt.x + eye_size//2, pt.y + eye_size//2, eye_size, eye_size))
        pygame.draw.rect(self.screen, constants.GREEN, pygame.Rect(self.food.x, self.food.y,constants.BLOCK_SIZE,
                                                                   constants.BLOCK_SIZE))  # food
        text = self.font.render("Score: " + str(self.score), True, constants.WHITE)
        self.screen.blit(text, [0, 0])
        pygame.display.flip()

    def __shortest_dist(self):
        shortest_dist = (self.head.x - self.food.x) ** 2 + (self.head.y - self.food.y) ** 2
        return shortest_dist ** 0.5

    # ai stuff
    def __change_ai_direction(self, ai_action):
        ai_action = ai_action.astype(int)
        if np.array_equal(ai_action, [1, 0, 0]):
            pass
        elif np.array_equal(ai_action, [0, 1, 0]):  # clockwise
            self.direction = self.__clockwise_turn()
        elif np.array_equal(ai_action, [0, 0, 1]):  # anticlockwise
            self.direction = self.__anti_clockwise_turn()

    def __clockwise_turn(self):
        clockwise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        return clockwise[(clockwise.index(self.direction) + 1) % len(clockwise)]

    def __anti_clockwise_turn(self):
        clockwise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        return clockwise[(clockwise.index(self.direction) - 1) % len(clockwise)]
