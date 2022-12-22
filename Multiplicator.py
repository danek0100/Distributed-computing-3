### LB #3
### Соболев Данил, Ольга Фролова, Диана Шарибжинова 19ПМИ-1

import matplotlib.pyplot as plt
from matplotlib.patches import Arc

import pandas as pd
import numpy as np
import celluloid as cld


def make_movie(df: pd.DataFrame, filename: str):
    camera = cld.Camera(plt.figure())
    plt.title(f'Animation from 0 to 1 period with step 0.001 step')

    for step in range(len(data.t)):
        ax = plt.gca().set_aspect('equal')
        keys_ = data.keys()
        x = [keys_[x] for x in range(1, len(keys_), 2)]
        y = [keys_[x] for x in range(2, len(keys_), 2)]
        df_x = []
        df_y = []
        for i in range(len(x)-1):
            df_x.append(df[x[i]][step])
        for i in range(len(y)):
            df_y.append(df[y[i]][step])
        plt.scatter(x=df_x, y=df_y, c='b', s=80)
        camera.snap()
    anim = camera.animate(blit=True)
    anim.save(filename)


data = pd.read_csv('outputSTR.txt', sep=',')

data['t'].astype(int)


make_movie(data, 'animation.gif')

