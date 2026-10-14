import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt


def plot_sections(section_times, coeffs, file_name, n_comp=1):
    """Function for plotting section times demo

    Parameters
    ----------
    section_times : ndarray, shape (m+1,)
        Polynomial breakpoints for piecewise polynomials.
    coeffs : ndarray, shape (k, m, …)
        Polynomial coefficients, order k and m intervals.
    file_name : string
        name for saving figure.
    """
    n_sec = int(len(coeffs)/n_comp)
    time = np.linspace(section_times[0], section_times[-1], 1001)

    plt.figure()

    y_max = 0
    for n in range(n_comp):
        start = n * n_sec
        end = (n + 1) * n_sec
        sections = scipy.interpolate.PPoly(coeffs[start:end], section_times)
        plt.plot(time, sections(time))

        y_max = max(y_max, max(sections(time)))

    sections_center = []
    for i in range(len(section_times)-1):
        sections_center.append((section_times[i] + section_times[i+1])/2)


    plt.vlines(section_times, 0, 1.1*y_max, 'k', '--')

    plt.ylabel('concentration')
    plt.xlabel('time')

    plt.show()

    if file_name is not None:
        plt.savefig(file_name)



if __name__ == '__main__':
    plt.close('all')
    section_times = [0, 1]
    coeffs = [[1]]
    file_name = './const.png'
    plot_sections(section_times, coeffs, file_name)

    section_times = [0, 1, 2]
    coeffs = [[1, 0]]
    file_name = './step.png'
    plot_sections(section_times, coeffs, file_name)

    section_times = [0, 1, 2, 3]
    coeffs = [[1, 0,-1],
              [0, 1, 1]]
    file_name = './gradient.png'
    plot_sections(section_times, coeffs, file_name)

    section_times = [0, 1, 2]
    coeffs = [[1, 0],
              [0, 0],
              [0, 1]]
    file_name = './quad.png'
    plot_sections(section_times, coeffs, file_name)

    section_times = [0, 1, 2]
    coeffs = [[0, 2],
              [0.5, 1]]
    file_name = './2_comp.png'
    plot_sections(section_times, coeffs, file_name, n_comp=2)
