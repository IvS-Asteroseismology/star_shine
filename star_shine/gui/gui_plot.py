"""STAR SHINE
Satellite Time-series Analysis Routine using Sinusoids and Harmonics through Iterative Non-linear Extraction

This Python module contains plotting functions for the graphical user interface.

Code written by: Luc IJspeert
"""
import os
import matplotlib as mpl
import matplotlib.figure
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT

from PySide6.QtWidgets import QWidget, QVBoxLayout
from PySide6.QtCore import Signal

from star_shine.config.helpers import get_images_path


class PlotToolbar(NavigationToolbar2QT):
    """New plot toolbar"""

    click_icon_file = os.path.join(get_images_path(), 'click')
    residual_icon_file = os.path.join(get_images_path(), 'residual')

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
        ('Residual', 'Show residual plot', residual_icon_file, 'residual'),
        ('Subplots', 'Configure subplots', 'subplots', 'configure_subplots'),
        ('Customize', 'Edit axis, curve and image parameters', 'qt4_editor_options', 'edit_parameters'),
        (None, None, None, None),
        ('Save', 'Save the figure', 'filesave', 'save_figure')
    ]

    def __init__(self, canvas, parent):
        super().__init__(canvas, parent)
        self._setup_click_button()
        self._setup_residual_button()

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

    def _setup_residual_button(self):
        """Set up the residual mode button."""
        for action in self.actions():
            if action.text() == "Residual":
                action.setCheckable(True)
                self.residual_action = action
                break

        return None

    def residual(self, *args):
        """Toggle residual mode."""
        if args and args[0]:
            self.residual_action.setChecked(True)
            return None
        elif args and not args[0]:
            self.residual_action.setChecked(False)
            return None

        return None


class PlotWidget(QWidget):
    """A widget for displaying plots using Matplotlib in a Qt application.

    Attributes
    ----------
    """
    # Define a signal that emits the plot ID and clicked coordinates
    click_signal = Signal(float, float, int)
    residual_signal = Signal()

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
        # self.figure.patch.set_facecolor('grey')
        # self.figure.patch.set_alpha(0.0)

        # Add toolbar for interactivity
        self.toolbar = PlotToolbar(self.canvas, self)
        self.show_residual = False

        # make an axis and set the labels
        self.ax = self.figure.add_subplot(111)
        self._set_labels()

        # make the layout and add the toolbar and canvas widgets
        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        # Connect the mouse click event to a custom method
        self.canvas.mpl_connect('button_press_event', self.on_click)

        # Connect the residual action to the on_residual method
        self.toolbar.residual_action.triggered.connect(self.on_residual)

        # plot types and properties supported
        self.plot_type_list = ['plot', 'scatter', 'vlines']
        self.plot_property_list = ['_xs', '_ys', '_labels', '_colors']
        self.property_fill_values = {'_xs': [], '_ys': [], '_labels': '', '_colors': None}

        # make references for the plot data and plot elements
        for plot_type in self.plot_type_list:
            setattr(self, plot_type + '_art', [])
            for plot_property in self.plot_property_list:
                key = plot_type + plot_property
                setattr(self, key, [])

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

        return None

    def on_click(self, event):
        """Click event"""
        # Left mouse button click
        if self.toolbar.click_action.isChecked() and event.button == 1:
            x, y = event.xdata, event.ydata
            # Ensure valid coordinates
            if x is not None and y is not None:
                self.click_signal.emit(x, y, event.button)

        # Right mouse button click
        if self.toolbar.click_action.isChecked() and event.button == 3:
            x, y = event.xdata, event.ydata
            # Ensure valid coordinates
            if x is not None and y is not None:
                self.click_signal.emit(x, y, event.button)

        return None

    def on_residual(self, event):
        """Residual event"""
        self.show_residual = self.toolbar.residual_action.isChecked()
        self.residual_signal.emit()

        return None

    def set_plot_data(self, **kwargs):
        """Set the plot data, wiping existing data.

        Keywords:
        line_xs, line_ys, line_labels, line_colors,
        scatter_xs, scatter_ys, scatter_labels, scatter_colors,
        vlines_xs, vlines_ys, vlines_labels, vlines_colors

        All arguments must be supplied as lists. Data is only set if `<type>_xs` and `<type>_ys`
        for the plot type is given. Lists per plot type are assumed of equal length.
        """
        # see if there is data in kwargs for any of the plot types
        for plot_type in self.plot_type_list:
            # go through the supplied and existing plot properties to check lengths
            n_existing = 0
            n_supplied = 0
            for plot_property in self.plot_property_list:
                key = plot_type + plot_property
                n_existing = max(n_existing, len(getattr(self, key)))  #  take the max length
                if key in kwargs.keys():
                    n_supplied = max(n_supplied, len(kwargs[key]))  #  take the max length

            # positive if more data supplied than existing plots
            n_diff = n_supplied - n_existing

            # go through the plot properties to set them accordingly
            for plot_property in self.plot_property_list:
                key = plot_type + plot_property
                if key in kwargs.keys() and (len(kwargs[key]) == n_supplied):
                    # set the plot property to the supplied value(s)
                    setattr(self, key, kwargs[key])
                elif key in kwargs.keys():
                    # set the plot property to the supplied value(s) and extend them with fill values
                    setattr(self, key, kwargs[key])
                    n_diff_i = n_supplied - len(kwargs[key])
                    getattr(self, key).extend([self.property_fill_values[plot_property] for _ in range(n_diff_i)])
                elif n_existing == 0:
                    # set the plot property to its fill value(s)
                    setattr(self, key, [self.property_fill_values[plot_property] for _ in range(n_supplied)])
                elif n_existing < n_supplied:
                    # extend the plot property with its fill value(s)
                    getattr(self, key).extend([self.property_fill_values[plot_property] for _ in range(n_diff)])
                else:  # n_existing >= n_supplied
                    # remove excess plot properties
                    setattr(self, key, getattr(self, key)[:n_supplied])

        return None

    @staticmethod
    def _update_plot_element(plot_type, plot_element, **kwargs):
        """Update an existing plot element."""
        # update the data
        if plot_type == 'plot':
            plot_element.set_xdata(kwargs['x'])
            plot_element.set_ydata(kwargs['y'])
        elif plot_type == 'scatter':
            plot_element.set_offsets(np.column_stack((kwargs['x'], kwargs['y'])))
        else:  # plot_type == 'vlines'
            plot_element.set_segments([[[xi, 0], [xi, yi]] for xi, yi in zip(kwargs['x'], kwargs['y'])])

        # update colour
        if 'color' in kwargs and kwargs['color'] is not None:
            if plot_type == 'plot':
                plot_element.set_color(kwargs['color'])
            elif plot_type == 'scatter':
                plot_element.set_facecolors(kwargs['color'])
            else:  # plot_type == 'vlines'
                plot_element.set_colors(kwargs['color'])

        # update label
        if 'label' in kwargs and kwargs['label'] != '':
            plot_element.set_label(kwargs['label'])

        return

    def _create_plot_element(self, plot_type, **kwargs):
        """Create a new plot element."""
        # make sure mpl gets the right keywords
        mpl_kwargs = {}
        if 'color' in kwargs and kwargs['color'] is not None:
            if plot_type == 'plot':
                mpl_kwargs['color'] = kwargs['color']
            elif plot_type == 'scatter':
                mpl_kwargs['c'] = kwargs['color']
            else:  # plot_type == 'vlines'
                mpl_kwargs['colors'] = kwargs['color']

        if 'label' in kwargs and kwargs['label'] != '':
            mpl_kwargs['label'] = kwargs['label']

        if plot_type == 'plot':
            art = self.ax.plot(kwargs['x'], kwargs['y'], **mpl_kwargs)[0]  # returns list
        elif plot_type == 'scatter':
            art = self.ax.scatter(kwargs['x'], kwargs['y'], marker='.', **mpl_kwargs)
        else:  # plot_type == 'vlines'
            ymin = np.zeros_like(kwargs['x'])
            art = self.ax.vlines(kwargs['x'], ymin=ymin, ymax=kwargs['y'], **mpl_kwargs)

        return art

    def update_plot(self):
        """Update the plot in the widget."""
        # update the plot (with altered or appended data)
        for plot_type in self.plot_type_list:
            plot_elements = getattr(self, plot_type + '_art')
            xs = getattr(self, plot_type + '_xs')
            ys = getattr(self, plot_type + '_ys')
            colors = getattr(self, plot_type + '_colors')
            labels = getattr(self, plot_type + '_labels')

            # for each plot element: if the element exists, update it, else create it
            for i in range(len(xs)):
                kwargs = {'x': xs[i], 'y': ys[i], 'color': colors[i], 'label': labels[i]}

                if i < len(plot_elements):
                    # update the plot element
                    self._update_plot_element(plot_type, plot_elements[i], **kwargs)
                else:
                    # create new plot element
                    art = self._create_plot_element(plot_type, **kwargs)
                    plot_elements.append(art)

        # Redraw the canvas to reflect changes
        # self.ax.legend()
        self.figure.tight_layout()  # needs to happen before draw
        self.canvas.draw()

        return None

    def new_plot(self):
        """Plot the data in the widget."""
        # Start by clearing the canvas
        self.clear_plot()
        self.plot_art = []  # reset all plot elements
        self.scatter_art = []  # reset all scatter elements
        self.vlines_art = []  # reset all vlines elements

        # Plot the line plot(s)
        self.update_plot()

        return None
