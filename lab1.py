import time

from scipy.integrate import solve_ivp
import numpy as np
from math import sin, cos, pi, sqrt
from matplotlib import pyplot as plt


def st_time(func):
    """
        st decorator to calculate the total time of a func
    """
    def st_func(*args, **keyArgs):
        t1 = time.time()
        r = func(*args, **keyArgs)
        t2 = time.time()
        print(f"Time={t2 - t1}")
        return r

    return st_func


# Ballistic coefficient
C = 0.15

ρ_dict = {  # kg/m^3
    'lead': 11340,
    'iron': 7874,
    'aluminium': 2699,
    'air': 1.29,
    'zero': 0
}


class Sphere:
    def __init__(self, diameter, ρ, env_ρ=ρ_dict['air']):
        self.diameter = diameter
        if type(ρ) == str:
            self.ρ = ρ_dict[ρ]
        else:
            self.ρ = ρ
        if type(env_ρ) == str:
            env_ρ = ρ_dict[env_ρ]
        self.radius = diameter / 2
        self.S = pi * (self.radius ** 2)
        self.V = (4 / 3) * pi * (self.radius ** 3)
        self.m = self.ρ * self.V
        self.β = C * env_ρ * self.S / 2

    def __str__(self):
        return f'Sphere D={self.diameter}; m={self.m}; S={self.S}; V={self.V}; β={self.β}'


class Env:
    def __init__(self, f):
        self.f = f
        self.α = pi / 4  # Rad
        self.interval = (0, 100)

        self.g = 9.8  # m/sec^2
        self.ε = 1

        # Starting speed
        self.v0 = 300
        self.u0 = self.v0 * cos(self.α)
        self.w0 = self.v0 * sin(self.α)
        self.x0 = 0
        self.y0 = 0

    def __call__(self, *args, **kwargs):
        return self.f(self, *args, **kwargs)


@Env
def x(env, t):
    return env.v0 * cos(env.α) * t


@Env
def y(env, t):
    return env.v0 * sin(env.α) * t - env.g * (t ** 2) / 2


def right_sys_part_init(func, item):
    def wrapper(t, system):
        return func(item, t, system)

    return wrapper


# Newton model
# decorated by right_sys_part_init
@Env
def right_sys_part(env, item, _, system):
    (u, w, x, y) = system
    fac = -item.β * sqrt(u ** 2 + w ** 2) / item.m
    return np.ndarray((4,), buffer=np.array([u * fac, w * fac - env.g, u, w]))


@Env
def init_system(env):
    return np.ndarray((4,), buffer=np.array([env.u0, env.w0, env.x0, env.y0]))


@st_time
@Env
def go_newton(env, item):
    return solve_ivp(right_sys_part_init(right_sys_part, item), env.interval, init_system(), max_step=env.ε)


def trim(arr):
    t = np.where(arr[1] >= 0)[-1][-1]
    return arr[:t, :t]


@st_time
def go_galileo(ts):
    xs = [x(t) for t in ts]
    ys = [y(t) for t in ts]
    return trim(np.ndarray((2, len(xs)), buffer=np.array([xs, ys])))


if __name__ == '__main__':
    # timeit.timeit("go_newton(Sphere(diameter=0.5, ρ='aluminium'))", setup="from __main__ import go_newton, Sphere")
    # exit(0)
    item_3 = Sphere(diameter=0.1, ρ='aluminium')
    item_2 = Sphere(diameter=0.25, ρ='aluminium')
    item_1 = Sphere(diameter=0.5, ρ='aluminium')
    print(item_1)
    print(item_2)
    print(item_3)

    coords_1 = go_newton(item_1)
    coords_2 = go_newton(item_2)
    coords_3 = go_newton(item_3)

    ts = coords_1['t']
    coords_1 = trim(coords_1['y'][2:])
    coords_2 = trim(coords_2['y'][2:])
    coords_3 = trim(coords_3['y'][2:])

    newton_xs, newton_ys = coords_1
    plt.plot(newton_xs, newton_ys, 'r', label=f'Newton D={item_1.diameter}')
    newton_xs, newton_ys = coords_2
    plt.plot(newton_xs, newton_ys, 'g', label=f'Newton D={item_2.diameter}')
    newton_xs, newton_ys = coords_3
    plt.plot(newton_xs, newton_ys, 'b', label=f'Newton D={item_3.diameter}')

    galileo_xs, galileo_ys = go_galileo(ts)

    print("Galileo:\t(", galileo_xs[-1], ", ", galileo_ys[-1], ")", sep='')
    print("Newton: \t(", newton_xs[-1], ", ", newton_ys[-1], ")", sep='')

    plt.plot(galileo_xs, galileo_ys, 'y', label='Galileo')
    # plt.plot(newton_xs, newton_ys, 'b', label='Newton')

    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    # plt.axis([0, 10000, 0, 5000])
