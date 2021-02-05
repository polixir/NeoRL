import numbers
import numpy as np
from gym import spaces
from neorl import core


NORTH = 1
EAST = 2
SOUTH = -1
WEST = -2
DIREC_DICT = {
    NORTH: "N",
    EAST: "E",
    SOUTH: "S",
    WEST: "W",
}


class LogisticsDistributionEnv(core.EnvData):
    def __init__(self, length=100, grid=3, point_num=3, speed_max=5):
        self.LENGTH = length
        self.GRID = grid
        self.POINT_NUM = point_num
        self.SPEED_MAX = speed_max

        self.observation_space = spaces.Box(
            low=-2, high=np.inf,
            shape=(length * length + 3,),  # flatten obs + speed + direction +SPEED_MAX
            dtype=np.int8)
        self.action_space = spaces.Discrete(length * length)

        self.MAP = np.zeros([length, length], dtype=np.int8)
        for i in range(length):
            for j in range((length - 1) // grid + 1):
                self.MAP[i, j * grid] = 1
                self.MAP[j * grid, i] = 1

        self.x, self.y = None, None
        self.speed = None
        self.direction = None
        self.points = None
        self.unfinished = None
        self.reset()

    def reset(self):
        self.x, self.y = self._roll_a_point()

        self.step_counter = 0
        self.speed = 0
        self.direction = NORTH
        self.points = {}

        # scatter cities
        while len(self.points) < self.POINT_NUM:
            self.points[self._roll_a_point()] = 1
        self.unfinished = self.POINT_NUM
        return self._draw_city()

    def step(self, action: numbers.Integral):
        target_x, target_y = action // self.LENGTH, action % self.LENGTH

        if (target_x % self.GRID == 0) or (target_y % self.GRID == 0):
            rew = self._reach_place(target_x, target_y)
        else:
            # trap
            rew = 100

        obs = self._draw_city()
        self.step_counter += 1

        done = (self.unfinished == 0) or (self.step_counter > self.LENGTH * self.LENGTH * self.POINT_NUM)

        return obs, -1 * rew, done, {"speed": self.speed, "direction": DIREC_DICT[self.direction]}

    def close(self):
        pass

    def render(self, mode='human'):
        pass

    def seed(self, seed=None):
        # important when use SubprocVectorEnv
        np.random.seed(seed)

    def _draw_city(self):
        obs = np.copy(self.MAP)
        for point in self.points:
            if self.points[point]:
                a, b = point
                obs[a, b] = 2
        obs[self.x, self.y] = 3

        return np.array([*obs.flatten(),
                         self.speed,
                         self.direction,
                         self.SPEED_MAX], dtype=np.int)

    def _roll_a_point(self):
        a = np.random.randint(self.LENGTH)
        b = np.random.randint(self.LENGTH)
        if np.random.randint(2):
            a -= a % self.GRID
        else:
            b -= b % self.GRID
        while (a, b) == (self.x, self.y):
            # if city generated where agent is. Re-do
            a, b = self._roll_a_point()
        return a, b

    def _same_street(self, m, n):
        return m == n and m % self.GRID == 0 and n % self.GRID == 0

    def _same_direction(self, m, n):
        return m % self.GRID == 0 and n % self.GRID == 0

    def _dash(self, a, b):
        assert self._same_street(self.x, a) or self._same_street(self.y, b)

        ret_time = 0

        if self.x == a and self.y == b:
            # stay punishment
            ret_time += 10
        else:
            if self.y < b:
                new_direction = EAST
            elif self.y > b:
                new_direction = WEST
            elif self.x < a:
                new_direction = SOUTH
            elif self.x > a:
                new_direction = NORTH
            else:
                assert 0
            dist = abs(self.y - b) + abs(self.x - a)
            if self.direction == new_direction:
                pass
            elif self.direction == - new_direction:
                self.speed = 0
            else:
                self.speed = self.speed / 2
            self.direction = new_direction
            s_tomax = (self.SPEED_MAX * self.SPEED_MAX - self.speed * self.speed) / 2
            if dist <= s_tomax:
                ret_time += np.sqrt(2 * dist + self.speed * self.speed) - self.speed
                self.speed += ret_time
            else:
                ret_time += self.SPEED_MAX - self.speed + (dist - s_tomax) / self.SPEED_MAX
                self.speed = self.SPEED_MAX

            self.x, self.y = a, b

        for point in self.points:
            if self.points[point]:
                if (a == point[0]) and (b == point[1]):
                    self.points[point] = 0
                    self.unfinished -= 1
                    ret_time -= 10
        return ret_time

    def _find_inter(self, m, n):
        if m // self.GRID == n // self.GRID:
            if m % self.GRID + n % self.GRID < self.GRID:
                ret = m - m % self.GRID
            else:
                ret = m - m % self.GRID + self.GRID
        else:
            if m > n:
                ret = m - m % self.GRID
            else:
                ret = m - m % self.GRID + self.GRID
        return ret

    def _reach_place(self, a, b):
        rew = 0
        if self._same_street(self.x, a) or self._same_street(self.y, b):
            rew += self._dash(a, b)
        elif self._same_direction(self.x, b):
            rew += self._dash(self.x, b)
            rew += self._dash(a, b)
        elif self._same_direction(a, self.y):
            rew += self._dash(a, self.y)
            rew += self._dash(a, b)
        elif self._same_direction(self.x, a):
            tmp = self._find_inter(self.y, b)
            rew += self._dash(self.x, tmp)
            rew += self._dash(a, tmp)
            rew += self._dash(a, b)
        elif self._same_direction(self.y, b):
            tmp = self._find_inter(self.x, a)
            rew += self._dash(tmp, self.y)
            rew += self._dash(tmp, b)
            rew += self._dash(a, b)
        else:
            assert 0

        return rew


def create_env():
    env = LogisticsDistributionEnv(length=8, grid=3, point_num=3)
    return env
