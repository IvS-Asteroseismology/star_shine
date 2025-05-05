"""STAR SHINE
Satellite Time-series Analysis Routine using Sinusoids and Harmonics through Iterative Non-linear Extraction

This Python module contains plotting functions for the graphical user interface.

Code written by: Luc IJspeert
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout
from PySide6.QtCore import Signal

import matplotlib as mpl
import matplotlib.figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as PlotToolbar


class PlotWidget(QWidget):
    """A widget for displaying plots using Matplotlib in a Qt application.

    Attributes
    ----------
    """
    # Define a signal that emits the plot ID and clicked coordinates
    click_signal = Signal(float, float)

    def __init__(self, title='Plot', xlabel='x', ylabel='y'):
        """A widget for displaying plots using Matplotlib in a Qt application.

        Parameters
        ----------
        title: str, optional
            Title of the plot. Default is 'Plot'.
        xlabel: str, optional
            Label for the x-axis. Default is 'x'.
        ylabel: str, optional
            Label for the y-axis. Default is 'y'.
        """
        super().__init__()
        # store some info
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel

        # set up the figure and canvas with an axis
        self.figure = mpl.figure.Figure()
        self.canvas = FigureCanvas(self.figure)
        self.figure.patch.set_facecolor('grey')
        self.figure.patch.set_alpha(0.0)

        # Add toolbar for interactivity
        self.toolbar = PlotToolbar(self.canvas, self)

        self.ax = self.figure.add_subplot(111)
        self._set_labels()

        # make the layout and add the canvas widget
        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        # Connect the mouse click event to a custom method
        self.canvas.mpl_connect('button_press_event', self.on_click)

        # Initialize color cycle
        self.color_cycler = iter(mpl.rcParams['axes.prop_cycle'].by_key()['color'])

    def _set_labels(self):
        """Set the axes labels and title."""
        self.ax.set_title(self.title)
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)

        return None

    def clear_plot(self):
        """Clear the plot"""
        self.ax.clear()

        # re-apply some elements
        self._set_labels()

        # Reset color cycler
        self.color_cycler = iter(mpl.rcParams['axes.prop_cycle'].by_key()['color'])

        return None

    def on_click(self, event):
        """Click event"""
        # Left mouse button click
        if event.button == 1:
            x, y = event.xdata, event.ydata
            # Ensure valid coordinates
            if x is not None and y is not None:
                self.click_signal.emit(x, y)

        return None

    def plot(self, x, y, **kwargs):
        """Plot a line graph on the widget.

        Parameters
        ----------
        x: array-like
            Data for the x-axis.
        y: array-like
            Data for the y-axis.
        **kwargs: dict, optional
            Additional keyword arguments to pass to matplotlib's plot function.
        """
        # get colour from the cycler
        color = next(self.color_cycler)

        # plot the thing
        self.ax.plot(x, y, c=color, **kwargs)

        # fix layout and draw
        self.figure.tight_layout()
        self.canvas.draw()

        return None

    def scatter(self, x, y, marker='.', **kwargs):
        """Plot a scatter graph on the widget.

        Parameters
        ----------
        x: array-like
            Data for the x-axis.
        y: array-like
            Data for the y-axis.
        marker: str
            Matplotlib marker keyword.
        **kwargs: dict, optional
            Additional keyword arguments to pass to matplotlib's scatter function.
        """
        # get colour from the cycler
        color = next(self.color_cycler)

        # plot the thing
        self.ax.scatter(x, y, c=color, marker=marker, **kwargs)

        # fix layout and draw
        self.figure.tight_layout()
        self.canvas.draw()

        return None
