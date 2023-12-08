from gym import Env
import numpy as np
from gym.spaces import Box
from sklearn.metrics import r2_score
from tensorflow.keras.models import load_model
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

class SystemRequest(BaseModel):
    name: str
    modelParams: list[float] = []


class SystemResponse(BaseModel):
    expected: list[list[float]] = []
    predicted: list[list[float]] = []
    r2Score: float

origins = [
    "http://localhost",
    "http://localhost:5173",
]

app = FastAPI(title='szakdoga')

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
)

@app.get('/')
async def home() -> SystemRequest:
    return SystemRequest(name='asd')

@app.post('/predict/')
async def predict(request: SystemRequest) -> SystemResponse:
    if(request.name == "Mass Spring Damper"):
        expected, predicted = MSDPredict(request.modelParams)
    elif(request.name == "Inverted Pendulum"):
        expected, predicted = IPPredict(request.modelParams)
    else:
        expected, predicted = []
    
    r2 = (r2_score(expected[0], predicted[0]) + r2_score(expected[1], predicted[1])) / 2
            
    return SystemResponse(expected=expected, predicted=predicted, r2Score=r2)



#region MSD system
from scipy.integrate import solve_ivp
def MassSpringDamper(t, y, a, b, c):
    dxdt = np.zeros(2)

    dxdt[0] = y[1]
    dxdt[1] = -(c / a) * y[0] - (b / a) * y[1]
    return dxdt



def createMatrix(i, M, position):
    temp = []
    for j in range(M):
        if i + j <= len(position):
            temp.append(position[i + j])
    return temp


def modifyMatrix(DD):
    temp = []
    for d in DD:
        temp.append(d[-1])
        del d[-1]
    return temp

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

frequency = 50

modelParams = [0.1, 0.1, 0.2]

tStart = float(0)
tEnd = float(20)

t = np.linspace(0, 20, frequency)

x0 = [1, 0]
sol = solve_ivp(MassSpringDamper, [tStart, tEnd], x0, t_eval=t, args=(modelParams))
times = sol.t
position = sol.y[0]

normalizedPosition = NormalizeData(position)
normalizedVelocity = NormalizeData(sol.y[1])

#endregion

#region Environment
class SzakdogaEnv(Env):
    def __init__(self, bemenet):
        self.action_space = Box(low=np.array([-1]), high=np.array([1]),shape=(1,))
        self.bemenet = bemenet

        self.current_step = 0
        self.max_steps = len(bemenet)
        self.state = 0 *np.zeros(self.max_steps)

        high = 1 * np.ones(self.max_steps)
        low = 0 * np.ones(self.max_steps)
        
        self.observation_space = Box(low=low, high=high)



    def step(self, action):
        difference = -1*(abs(self.bemenet[self.current_step] - action))

        for i in range(0,len(self.state)-1):
            self.state[i]=self.state[i+1]
        self.state[-1]=action

        info = {}

        self.current_step += 1

        if self.current_step==self.max_steps:
            done = True
        else:
            done = False

        if np.isnan(difference) or np.isinf(difference):
          difference = -np.finfo(np.float32).max
          done = True

        return self.state, difference, done, info

    def render(self, mode='human'):
        pass


    def reset(self):
        self.current_step = 0
        self.state = 0 *np.zeros(self.max_steps)
        return self.state
#endregion


def MSDPredict(params):
    modelMSDy0 = load_model("./models/MSD-0024-y0[50]/MSD-0024-y0[50]")
    modelMSDy1 = load_model("./models/MSD-0029-y1[50]/MSD-0029-y1[50]")

    env1 = SzakdogaEnv(normalizedPosition)
    obs1 = env1.reset()
    done1 = False
    actions1 = []

    while not done1:
        action1 = modelMSDy0.predict(np.expand_dims(obs1, axis=0))
        action1 = np.squeeze(action1)
        obs1, rewards, done1, _ = env1.step(action1)
        actions1.append(action1)

    env2 = SzakdogaEnv(normalizedVelocity)
    obs2 = env2.reset()
    done2 = False
    actions2 = []

    while not done2:
        action2 = modelMSDy1.predict(np.expand_dims(obs2, axis=0))
        action2 = np.squeeze(action2)
        obs2, rewards, done2, _ = env2.step(action2)
        actions2.append(action2)

    expected = solve_ivp(MassSpringDamper, [tStart, tEnd], x0, t_eval=t, args=params).y

    predicted = [actions1, actions2]
    return ([NormalizeData(expected[0]),NormalizeData(expected[1])], predicted)

#region IP system
from scipy.integrate import solve_ivp
import math
import numpy as np


def InvertedPendulum(t, x, a, b, c):
    g = a
    l = b
    b = c

    dxdt = np.zeros(2)

    dxdt[0] = x[1]
    dxdt[1] = (g / l) * math.sin(x[0]) - b * x[1]

    return dxdt


g = 9.81  # [m/s**2]
l = 1  # [m]
b = 1

vModelParams = [g, l, b]

tStart = 0
tEnd = 20

x0 = [0.01, 0]
t = np.linspace(0, 20, 50)
sol = solve_ivp(InvertedPendulum, [tStart, tEnd], x0, t_eval=t, args=vModelParams)

times = sol.t
position = sol.y[0]
speed = sol.y[1]

DD = []
for i in range(len(times) - 2):
    D = np.array([[times[i], times[i + 1], times[i + 2]],
                  [position[i], position[i + 1], position[i + 2]],
                  [speed[i], speed[i + 1], speed[i + 2]]])
    DD.append(D)


def NormalizeDataIP(data):
  min = np.min(data)
  max = np.max(data)
  for i in range(len(data)):
    data[i] = (data[i]- min) / (max - min)
  return data

normalizedPositionIP = NormalizeDataIP(sol.y[0])
normalizedVelocityIP = NormalizeDataIP(sol.y[1])

#endregion

def IPPredict(params):
    modelIPy0 = load_model("./models/IP-0004-y0[50]/IP-0004-y0[50]")
    modelIPy1 = load_model("./models/IP-0018-y1[50]-750/IP-0018-y1[50]-750")

    env1 = SzakdogaEnv(normalizedPositionIP)
    obs1 = env1.reset()
    done1 = False
    actions1 = []

    while not done1:
        action1 = modelIPy0.predict(np.expand_dims(obs1, axis=0))
        action1 = np.squeeze(action1)
        obs1, rewards, done1, _ = env1.step(action1)
        actions1.append(action1)

    env2 = SzakdogaEnv(normalizedVelocityIP)
    obs2 = env2.reset()
    done2 = False
    actions2 = []

    while not done2:
        action2 = modelIPy1.predict(np.expand_dims(obs2, axis=0))
        action2 = np.squeeze(action2)
        obs2, rewards, done2, _ = env2.step(action2)
        actions2.append(action2)

    expected = solve_ivp(InvertedPendulum, [tStart, tEnd], x0, t_eval=t, args=vModelParams).y

    predicted = [actions1, actions2]
    return ([NormalizeDataIP(expected[0]),NormalizeDataIP(expected[1])], predicted)