import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import streamlit.components.v1 as components


def update_line(num, data, line):
    line.set_data(data[..., :num])
    return (line,)


def run() -> None:
    fig = plt.figure()

    # Fixing random state for reproducibility
    np.random.seed(19680801)

    data = np.random.rand(2, 25)
    (l,) = plt.plot([], [], "r-")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("x")
    plt.title("test")
    line_ani = animation.FuncAnimation(
        fig, update_line, 25, fargs=(data, l), interval=200, blit=True
    )

    st.title("Embed Matplotlib animation in Streamlit")
    st.markdown("https://matplotlib.org/gallery/animation/basic_example.html")
    components.html(line_ani.to_jshtml(), height=1000)
