
from gymnasium import spaces
import numpy as np

REGISTERED_ROBOTS = {}

class Meta(type):
    def __new__(meta, name, bases, class_dict):
        cls = super().__new__(meta, name, bases, class_dict)

        # List all environments that should not be registered here.
        _unregistered_envs = []
        print(cls.__name__)
        if cls.__name__ not in _unregistered_envs:
            REGISTERED_ROBOTS[cls.__name__] = cls
        print("__new__")
        return cls

class RobotA(metaclass=Meta):
    pass

class RobotB(RobotA):
    pass

class RobotC(RobotB):
    pass

# Foo = type('Foo', (), {'name': 'kavin'})
# foo = Foo()
# print(foo.name)

print(REGISTERED_ROBOTS)
# print(REGISTERED_ROBOTS['RobotB']())
# print(RobotB.type)


high = np.inf * np.ones(3)
low = -high
observation_space = spaces.Box(low, high)
print(observation_space)
print(observation_space.low)
print(observation_space.high)
print(observation_space.shape[0])


obs_dict = {}
obs_dict['a'] = np.array([1, 2, 3])
obs_dict['b'] = np.array([4, 5, 6])
ob_lst = []
for key in obs_dict:
    ob_lst.append(np.array(obs_dict[key]).flatten())
print(ob_lst)
res = np.concatenate(ob_lst)
print(res)

