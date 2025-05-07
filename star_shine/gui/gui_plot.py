"""STAR SHINE
Satellite Time-series Analysis Routine using Sinusoids and Harmonics through Iterative Non-linear Extraction

This Python module contains plotting functions for the graphical user interface.

Code written by: Luc IJspeert
"""
import os
import matplotlib as mpl
import matplotlib.figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT

from PySide6.QtWidgets import QWidget, QVBoxLayout
from PySide6.QtCore import Signal

from star_shine.config.helpers import get_images_path


class PlotToolbar(NavigationToolbar2QT):
    """New plot toolbar"""

    click_icon_file = os.path.join(get_images_path(), 'click')

    # list of toolitems to add to the toolbar, format is:
    # (
    #   text, # the text of the button (often not visible to users)
    #   tooltip_text, # the tooltip shown on hover (where possible)
    #   image_file, # name of the image for the button (without the extension)
    #   name_of_method, # name of the method in NavigationToolbar2 to call
    # )
    NavigationToolbar2QT.toolitems = [
        ('Home', 'Reset original view', 'home', 'home'),
        ('Back', 'Back to previous view', 'back', 'back'),
        ('Forward', 'Forward to next view', 'forward', 'forward'),
        (None, None, None, None),
        ('Pan', 'Left button pans, Right button zooms\nx/y fixes axis, CTRL fixes aspect', 'move', 'pan'),
        ('Zoom', 'Zoom to rectangle\nx/y fixes axis', 'zoom_to_rect', 'zoom'),
        ('Click', 'Click on the plot to interact', click_icon_file, 'click'),
        (None, None, None, None),
        ('Subplots', 'Configure subplots', 'subplots', 'configure_subplots'),
        ('Customize', 'Edit axis, curve and image parameters', 'qt4_editor_options', 'edit_parameters'),
        (None, None, None, None),
        ('Save', 'Save the figure', 'filesave', 'save_figure')
    ]

    def __init__(self, canvas, parent):
        super().__init__(canvas, parent)
        self._setup_click_button()

    def _setup_click_button(self):
        """Set up the click mode button."""
        for action in self.actions():
            if action.text() == "Click":
                action.setCheckable(True)
                self.click_action = action
                break

        return None

    def click(self, *args):
        """Toggle click mode."""
        if args and args[0]:
            self.click_action.setChecked(True)
            return None
        elif args and not args[0]:
            self.click_action.setChecked(False)
            return None

        # handle state of built-in methods
        if not args and self.mode == 'pan/zoom':
            NavigationToolbar2QT.pan(self, False)
        elif not args and self.mode == 'zoom rect':
            NavigationToolbar2QT.zoom(self, False)

        return None

    def pan(self, *args):
        """Toggle pan mode."""
        # handle state of click button
        if self.click_action.isChecked():
            self.click(False)

        super().pan(*args)

    def zoom(self, *args):
        """Toggle zoom mode."""
        # handle state of click button
        if self.click_action.isChecked():
            self.click(False)

        super().zoom(*args)


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
        if self.toolbar.click_action.isChecked() and event.button == 1:
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
