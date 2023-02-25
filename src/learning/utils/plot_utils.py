"""
This file is part of Learned Inertial Model Odometry.
Copyright (C) 2023 Giovanni Cioffi <cioffi at ifi dot uzh dot ch>
(Robotics and Perception Group, University of Zurich, Switzerland).
This file is subject to the terms and conditions defined in the file
'LICENSE', which is part of this source code package.
"""

import matplotlib.pyplot as plt
import numpy as np


def xyPlot(title, labelx, labely, 
    vec1, label1, 
    vec2 = None, label2 = None, 
    vec3 = None, label3 = None,
    vec4 = None, label4 = None):
    plt.plot(vec1[:, 0], vec1[:, 1], label=label1)
    if vec2 is not None:
        plt.plot(vec2[:, 0], vec2[:, 1], label=label2)
    if vec3 is not None:
        plt.plot(vec3[:, 0], vec3[:, 1], label=label3)
    if vec4 is not None:
        plt.plot(vec4[:, 0], vec4[:, 1], label=label4)
    plt.grid()
    plt.legend()
    plt.xlabel(labelx)
    plt.ylabel(labely)
    plt.title(title)


def xyztPlot(title, 
    vec1, label1, 
    vec2 = None, label2 = None,
    vec3 = None, label3 = None,
    vec4 = None, label4 = None):
    plt.subplot(311)
    plt.plot(vec1[:,0], vec1[:,1], label=label1)
    if vec2 is not None:
        plt.plot(vec2[:,0], vec2[:,1], label=label2)
    if vec3 is not None:
        plt.plot(vec3[:,0], vec3[:,1], label=label3)
    if vec4 is not None:
        plt.plot(vec4[:,0], vec4[:,1], label=label4)
    plt.grid()
    plt.legend()
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title(title)

    plt.subplot(312)
    plt.plot(vec1[:,0], vec1[:,2], label=label1)
    if vec2 is not None:
        plt.plot(vec2[:,0], vec2[:,2], label=label2)
    if vec3 is not None:
        plt.plot(vec3[:,0], vec3[:,2], label=label3)
    if vec4 is not None:
        plt.plot(vec4[:,0], vec4[:,2], label=label4)
    plt.grid()
    plt.legend()
    plt.xlabel('t')
    plt.ylabel('y')

    plt.subplot(313)
    plt.plot(vec1[:,0], vec1[:,3], label=label1)
    if vec2 is not None:
        plt.plot(vec2[:,0], vec2[:,3], label=label2)
    if vec3 is not None:
        plt.plot(vec3[:,0], vec3[:,3], label=label3)
    if vec4 is not None:
        plt.plot(vec4[:,0], vec4[:,3], label=label4)
    plt.grid()
    plt.legend()
    plt.xlabel('t')
    plt.ylabel('z')

