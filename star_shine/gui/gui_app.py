"""STAR SHINE
Satellite Time-series Analysis Routine using Sinusoids and Harmonics through Iterative Non-linear Extraction

This Python module contains the graphical user interface.

Code written by: Luc IJspeert
"""
import os
import sys
import functools

from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QGridLayout, QSplitter, QMenuBar
from PySide6.QtWidgets import QWidget, QLabel, QTextEdit, QLineEdit, QSpinBox, QFileDialog, QMessageBox, QPushButton
from PySide6.QtWidgets import QTableView, QHeaderView, QDialog, QFormLayout, QCheckBox
from PySide6.QtGui import QAction, QFont, QScreen, QStandardItemModel, QStandardItem, QTextCursor, QIcon
from PySide6.QtGui import QDoubleValidator

from star_shine.core import utility as ut
from star_shine.api import Data, Result, Pipeline
from star_shine.gui import gui_log, gui_plot, gui_analysis, gui_config
from star_shine.config import helpers as hlp


# load configuration
config = hlp.get_config()


class MainWindow(QMainWindow):
    """The main window of the Star Shine application.

    Contains a graphical user interface for loading data, performing analysis,
    displaying results, and visualizing plots.
    """

    def __init__(self):
        super().__init__()

        # Get screen dimensions
        screen = QApplication.primaryScreen()
        screen_size = screen.availableSize()
        h_size = int(screen_size.width() * config.h_size_frac)  # some fraction of the screen width
        v_size = int(screen_size.height() * config.v_size_frac)  # some fraction of the screen height

        # Set some window things
        self.setWindowTitle("Star Shine")
        self.setGeometry(100, 50, h_size, v_size)  # x, y, width, height

        # App icon
        icon_path = os.path.join(hlp.get_images_path(), 'Star_Shine_dark_simple_small_transparent.png')
        self.setWindowIcon(QIcon(icon_path))

        # some things that need a default value
        self.data_dir = config.data_dir
        if self.data_dir == '':
            self.data_dir = os.path.expanduser('~')
        self.save_dir = config.save_dir
        if self.save_dir == '':
            self.save_dir = os.path.expanduser('~')
        self.save_subdir = ''

        # Set font size for the entire application
        font = QFont()
        font.setPointSize(11)
        QApplication.setFont(font)

        # Create the central widget and set the layout
        self._setup_central_widget()

        # Create the menu bar
        self._setup_menu_bar()

        # Add widgets to the layout
        self._add_widgets_to_layout()

        # custom gui-specific logger (will be reloaded and connected when data is loaded)
        self.logger = gui_log.get_custom_gui_logger('gui_logger', '')
        self.logger.log_signal.connect(self.append_text)

        # add the api classes for functionality
        self.pipeline = None
        self.pipeline_thread = None

    def _setup_central_widget(self):
        """Set up the central widget and its layout."""
        # Create a central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # create a horizontal layout
        self.layout = QHBoxLayout()
        self.central_widget.setLayout(self.layout)

        # create a splitter
        self.splitter = QSplitter()
        self.central_widget.layout().addWidget(self.splitter)

        return None

    def _add_widgets_to_layout(self):
        """Add widgets to the main window layout."""
        # Left column: input and text output field
        left_column = self._create_left_column()
        self.splitter.addWidget(left_column)

        # Middle column: frequencies
        middle_column = self._create_middle_column()
        self.splitter.addWidget(middle_column)

        # right column: plot area
        right_column = self._create_right_column()
        self.splitter.addWidget(right_column)

        # Set initial sizes for each column
        h_size = self.width()
        self.splitter.setSizes([h_size*3//9, h_size*2//9, h_size*4//9])

        return None

    def _setup_menu_bar(self):
        """Set up the menu bar with file and info menus."""
        menu_bar = QMenuBar()
        self.setMenuBar(menu_bar)

        # Add "File" menu and set it up
        file_menu = menu_bar.addMenu("File")
        self._setup_file_menu(file_menu)

        # Add "View" menu and set it up
        view_menu = menu_bar.addMenu("View")
        self._setup_view_menu(view_menu)

        # Add "Info" menu
        info_menu = menu_bar.addMenu("Info")
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about_dialog)
        info_menu.addAction(about_action)

        return None

    def _setup_file_menu(self, file_menu):
        """Set up the file menu."""

        # Add "Load Data" button to "File" menu
        load_data_action = QAction("Load Data", self)
        load_data_action.triggered.connect(self.load_data)
        file_menu.addAction(load_data_action)

        # Add "Save Data Object" button to "File" menu
        save_data_action = QAction("Save Data", self)
        save_data_action.triggered.connect(self.save_data)
        file_menu.addAction(save_data_action)

        # Add "Load Result Object" button to "File" menu
        load_result_action = QAction("Load Result", self)
        load_result_action.triggered.connect(self.load_result)
        file_menu.addAction(load_result_action)

        # Add "Save Result Object" button to "File" menu
        save_result_action = QAction("Save Result", self)
        save_result_action.triggered.connect(self.save_result)
        file_menu.addAction(save_result_action)

        # Add a horizontal separator
        file_menu.addSeparator()

        # Add "Set Save Location" button to "File" menu
        set_save_location_action = QAction("Set Save Location", self)
        set_save_location_action.triggered.connect(self.set_save_location)
        file_menu.addAction(set_save_location_action)

        # Add "Settings" action to open the settings dialog
        settings_action = QAction("Settings", self)
        settings_action.triggered.connect(self.show_settings_dialog)
        file_menu.addAction(settings_action)

        # Add a horizontal separator
        file_menu.addSeparator()

        # Add "Exit" button to "File" menu
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        return None

    def _setup_view_menu(self, view_menu):
        """Set up the view menu."""
        # Add a horizontal separator
        # view_menu.addSeparator()

        # Add "Save Layout" button to "View" menu
        self.save_layout_action = QAction("Save Layout", self)
        # self.save_layout_action.triggered.connect()
        view_menu.addAction(self.save_layout_action)

        return None

    def _create_left_column(self):
        """Create and return the left column widget.

        Returns
        -------
        QWidget
            The left column widget containing input fields, buttons, and text output.
        """
        # create a vertical layout for in the left column of the main layout
        l_col_widget = QWidget()
        l_col_layout = QVBoxLayout(l_col_widget)

        # create the info grid
        info_widget = self._create_info_fields()
        l_col_layout.addWidget(info_widget)

        # create a horizontal layout with some buttons
        run_button_widget = QWidget()
        run_button_layout = QHBoxLayout(run_button_widget)

        # Button for starting analysis
        self.analyze_button = QPushButton("Run Full Analyis")
        self.analyze_button.clicked.connect(functools.partial(self.perform_analysis, 'run'))
        run_button_layout.addWidget(self.analyze_button)

        # Button for starting analysis
        self.stop_button = QPushButton("Stop [WIP]")
        self.stop_button.clicked.connect(self.stop_analysis)
        run_button_layout.addWidget(self.stop_button)

        l_col_layout.addWidget(run_button_widget)

        # create a horizontal layout with the buttons for each step
        steps_button_widget = QWidget()
        steps_button_layout = QHBoxLayout(steps_button_widget)

        # Create a spin box for integer input
        self.spin_box = QSpinBox(self)
        self.spin_box.setRange(0, 99999)
        self.spin_box.setValue(0)
        # Button for starting iterative prewhitening
        self.extract_button = QPushButton("Extract")
        self.extract_button.clicked.connect(lambda: self.perform_analysis('iterative_prewhitening',
                                                                          n_extract=self.spin_box.value()))
        steps_button_layout.addWidget(self.extract_button)
        steps_button_layout.addWidget(self.spin_box)  # add the number field after the button

        # Button for starting sinusoid fit
        self.free_fit_button = QPushButton("Free Fit")
        self.free_fit_button.clicked.connect(functools.partial(self.perform_analysis, 'optimise_sinusoid'))
        steps_button_layout.addWidget(self.free_fit_button)

        # Button for starting finder
        self.find_harmonic_button = QPushButton("Find Harmonics")
        self.find_harmonic_button.clicked.connect(functools.partial(self.perform_analysis, 'couple_harmonics'))
        steps_button_layout.addWidget(self.find_harmonic_button)

        # Button for inputting
        self.add_harmonic_button = QPushButton("Add Base Harmonic")
        self.add_harmonic_button.clicked.connect(self.add_harmonic_series)
        steps_button_layout.addWidget(self.add_harmonic_button)

        # Button for saving the results
        self.save_result_button = QPushButton("Save Result")
        self.save_result_button.clicked.connect(self.save_result)
        steps_button_layout.addWidget(self.save_result_button)

        l_col_layout.addWidget(steps_button_widget)

        # create a horizontal layout with some buttons
        extra_button_widget = QWidget()
        extra_button_layout = QHBoxLayout(extra_button_widget)

        # Button for adding base harmonic frequency
        self.harmonic_button = QPushButton("Add base harmonic")
        # self.harmonic_button.clicked.connect(self.add_base_harmonic)
        extra_button_layout.addWidget(self.harmonic_button)

        l_col_layout.addWidget(extra_button_widget)

        # Log area
        log_label = QLabel("Log:")
        l_col_layout.addWidget(log_label)

        self.text_field = QTextEdit()
        self.text_field.setReadOnly(True)  # Make the text edit read-only
        l_col_layout.addWidget(self.text_field)

        return l_col_widget

    def _create_middle_column(self):
        """Create and return the middle column widget.

        Returns
        -------
        QWidget
            The middle column widget containing a table view.
        """
        # create a vertical layout for in the middle column of the main layout
        m_col_widget = QWidget()
        m_col_layout = QVBoxLayout(m_col_widget)

        # add the data model formula above the table
        equation_str = "Model: flux = \u2211\u1D62 (a\u1D62 sin(2\u03C0f\u1D62t + \u03C6\u1D62)) + bt + c"
        formula_label = QLabel(equation_str)
        m_col_layout.addWidget(formula_label)

        # Create the table view and model
        self.table_view = QTableView()
        self.table_model = QStandardItemModel(0, 3)  # Start with 0 rows and 3 columns
        self.table_model.setHorizontalHeaderLabels(["Frequency", "Amplitude", "Phase"])
        self.table_view.setModel(self.table_model)

        # Set the horizontal header's stretch mode for each column
        h_header = self.table_view.horizontalHeader()
        h_header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)  # Stretch all columns proportionally

        # Set the vertical header's stretch mode for each row
        v_header = self.table_view.verticalHeader()
        v_header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)

        # Add the button and table view to the middle column layout
        m_col_layout.addWidget(self.table_view)

        return m_col_widget

    def _create_right_column(self):
        """Create and return the right column widget.

        Returns
        -------
        QWidget
            The right column widget containing plot areas.
        """
        # create a vertical layout for in the right column of the main layout
        r_col_widget = QWidget()
        r_col_layout = QVBoxLayout(r_col_widget)

        # upper plot area for the data
        self.upper_plot_area = gui_plot.PlotWidget(title='Data', xlabel='time', ylabel='flux')
        r_col_layout.addWidget(self.upper_plot_area)

        # connect the residual event
        self.upper_plot_area.residual_signal.connect(functools.partial(self.update_plots, new_plot=True))

        # lower plot area for the periodogram
        self.lower_plot_area = gui_plot.PlotWidget(title='Periodogram', xlabel='frequency', ylabel='amplitude')
        r_col_layout.addWidget(self.lower_plot_area)

        # connect the click and residual event
        self.lower_plot_area.click_signal.connect(self.click_periodogram)
        self.lower_plot_area.residual_signal.connect(functools.partial(self.update_plots, new_plot=True))

        return r_col_widget

    def _create_info_fields(self):
        """Create and return the left column information grid widget.

        Returns
        -------
        QWidget
            The left column information grid widget.
        """
        # create a grid area for labels and read-only text fields
        info_widget = QWidget()
        info_layout = QGridLayout(info_widget)

        # current directory
        current_dir_label = QLabel("Current directory:")
        self.current_dir_field = QLineEdit()
        self.current_dir_field.setReadOnly(True)
        self.current_dir_field.setText(self.save_dir)
        info_layout.addWidget(current_dir_label, 0, 0)
        info_layout.addWidget(self.current_dir_field, 0, 1)

        # target id
        target_id_label = QLabel("Target ID:")
        self.target_id_field = QLineEdit()
        self.target_id_field.setReadOnly(True)
        info_layout.addWidget(target_id_label, 1, 0)
        info_layout.addWidget(self.target_id_field, 1, 1)

        # data id
        data_id_label = QLabel("Data ID:")
        self.data_id_field = QLineEdit()
        self.data_id_field.setReadOnly(True)
        info_layout.addWidget(data_id_label, 1, 2)
        info_layout.addWidget(self.data_id_field, 1, 3)

        return info_widget

    def update_info_fields(self):
        """Update the information field widget."""
        # current directory
        self.current_dir_field.setText(self.save_dir)

        # if there is a pipeline we can fill in the data attributes
        if self.pipeline is not None:
            self.target_id_field.setText(self.pipeline.data.target_id)
            self.data_id_field.setText(self.pipeline.data.data_id)

        return None

    def append_text(self, text):
        """Append a line of text at the end of the plain text output box.

        Parameters
        ----------
        text: str
            The text to append.
        """
        cursor = self.text_field.textCursor()
        cursor.movePosition(QTextCursor.End)  # Move cursor to the end of the text
        cursor.insertText(text + '\n')  # insert the text
        self.text_field.setTextCursor(cursor)
        self.text_field.ensureCursorVisible()

        return None

    def update_table(self, display_err=True):
        """Fill the table with the given data."""
        # get the result parameters
        col1 = self.pipeline.ts_model.sinusoid.f_n
        col2 = self.pipeline.ts_model.sinusoid.a_n
        col3 = self.pipeline.ts_model.sinusoid.ph_n

        # make sure uncertainties are updated
        self.pipeline.ts_model.update_sinusoid_uncertainties()
        col1_err = self.pipeline.ts_model.sinusoid.f_n_err
        col2_err = self.pipeline.ts_model.sinusoid.a_n_err
        col3_err = self.pipeline.ts_model.sinusoid.ph_n_err

        # display sinusoid parameters in the table
        self.table_model.setRowCount(len(col1))
        for row, row_items in enumerate(zip(col1, col2, col3, col1_err, col2_err, col3_err)):
            # convert to strings
            c1 = ut.float_to_str_scientific(row_items[0], row_items[3], error=display_err, brackets=False)
            c2 = ut.float_to_str_scientific(row_items[1], row_items[4], error=display_err, brackets=False)
            c3 = ut.float_to_str_scientific(row_items[2], row_items[5], error=display_err, brackets=False)

            # convert to table items
            c1_item = QStandardItem(c1)
            c2_item = QStandardItem(c2)
            c3_item = QStandardItem(c3)

            # insert into table
            self.table_model.setItem(row, 0, c1_item)
            self.table_model.setItem(row, 1, c2_item)
            self.table_model.setItem(row, 2, c3_item)

        return None

    def update_plots(self, new_plot=False):
        """Update the plotting area with the current data."""
        # collect plot data in a dict
        upper_plot_data = {}
        lower_plot_data = {}

        # upper plot area - time series
        upper_plot_data['scatter_xs'] = [self.pipeline.ts_model.time]
        upper_plot_data['scatter_ys'] = [self.pipeline.ts_model.flux]
        # lower plot area - periodogram
        lower_plot_data['plot_xs'] = [self.pipeline.ts_model.pd_freqs]
        lower_plot_data['plot_ys'] = [self.pipeline.ts_model.pd_ampls]

        # include result attributes if present
        if self.pipeline.result.target_id != '' and not self.upper_plot_area.show_residual:
            # upper plot area - time series
            upper_plot_data['plot_xs'] = [self.pipeline.ts_model.time]
            upper_plot_data['plot_ys'] = [self.pipeline.ts_model.full_model()]
            upper_plot_data['plot_colors'] = ['grey']

        if self.pipeline.result.target_id != '' and not self.lower_plot_area.show_residual:
            # lower plot area - periodogram
            freqs, ampls = self.pipeline.ts_model.periodogram(subtract_model=True)
            lower_plot_data['plot_xs'].append(freqs)
            lower_plot_data['plot_ys'].append(ampls)
            lower_plot_data['vlines_xs'] = [self.pipeline.ts_model.sinusoid.f_n]
            lower_plot_data['vlines_ys'] = [self.pipeline.ts_model.sinusoid.a_n]
            lower_plot_data['vlines_colors'] = ['grey']

        # only show residual if toggle checked
        if self.pipeline.result.target_id != '' and self.upper_plot_area.show_residual:
            # upper plot area - time series
            upper_plot_data['scatter_xs'] = [self.pipeline.ts_model.time]
            upper_plot_data['scatter_ys'] = [self.pipeline.ts_model.residual()]

        if self.pipeline.result.target_id != '' and self.lower_plot_area.show_residual:
            # lower plot area - periodogram
            freqs, ampls = self.pipeline.ts_model.periodogram(subtract_model=True)
            lower_plot_data['plot_xs'] = [freqs]
            lower_plot_data['plot_ys'] = [ampls]

        # set the plot data
        self.upper_plot_area.set_plot_data(**upper_plot_data)
        self.lower_plot_area.set_plot_data(**lower_plot_data)

        # start with a fresh plot
        if new_plot:
            self.upper_plot_area.new_plot()
            self.lower_plot_area.new_plot()
        else:
            self.upper_plot_area.update_plot()
            self.lower_plot_area.update_plot()

        return None

    def new_dataset(self, data):
        """Set up pipeline and logger for the new data that was loaded"""
        # for saving, make a folder if not there yet
        self.save_subdir = f"{data.target_id}_analysis"

        full_dir = os.path.join(self.save_dir, self.save_subdir)
        if not os.path.isdir(full_dir):
            os.mkdir(full_dir)  # create the subdir

        # custom gui-specific logger
        self.logger = gui_log.get_custom_gui_logger(data.target_id, full_dir)
        self.logger.log_signal.connect(self.append_text)

        # Make ready the pipeline class
        self.pipeline = Pipeline(data=data, save_dir=self.save_dir, logger=self.logger)

        # set up a pipeline thread
        self.pipeline_thread = gui_analysis.PipelineThread(self.pipeline)
        self.pipeline_thread.result_signal.connect(self.handle_result_signal)

        # update the info fields
        self.update_info_fields()

        # clear and update the plots
        self.update_plots(new_plot=True)

        return None

    def set_save_location(self):
        """Open a dialog to select the save location."""
        # Open a directory selection dialog
        new_dir = QFileDialog.getExistingDirectory(self, caption="Select Save Location", dir=self.save_dir)

        if new_dir:
            self.save_dir = new_dir
            self.append_text(f"Save location set to: {self.save_dir}")

        # update the current directory info field
        self.update_info_fields()

        return None

    def load_data(self):
        """Read data from a file or multiple files using a dialog window."""
        # get the path(s) from a standard file selection screen
        file_paths, _ = QFileDialog.getOpenFileNames(self, caption="Read Data", dir=self.save_dir,
                                                     filter="All Files (*)")

        # do nothing in case no file(s) selected
        if not file_paths:
            return None

        # load data into instance
        if len(file_paths) == 1 and file_paths[0].endswith('.hdf5'):
            # a single hdf5 file is loaded as a star shine data object
            data = Data.load(file_name=file_paths[0], data_dir='', logger=self.logger)
        else:
            # any other files are loaded as external data
            data = Data.load_data(file_list=file_paths, data_dir='', target_id='', data_id='', logger=self.logger)

        # set the save dir to the one where we opened the data
        self.save_dir = os.path.dirname(file_paths[0])

        # set up some things
        self.new_dataset(data)

        return None

    def save_data(self):
        """Save data to a file using a dialog window."""
        # check whether data is present
        if self.pipeline is None or len(self.pipeline.data.file_list) == 0:
            self.logger.error("Input Error: please load data first.")
            return None

        suggested_path = os.path.join(self.save_dir, self.pipeline.data.target_id + '_data.hdf5')
        file_path, _ = QFileDialog.getSaveFileName(self, caption="Save Data", dir=suggested_path,
                                                   filter="HDF5 Files (*.hdf5);;All Files (*)")

        # do nothing in case no file selected
        if not file_path:
            return None

        self.pipeline.data.save(file_path)

        return None

    def load_result(self):
        """Load result from a file using a dialog window."""
        # check whether a pipeline is present
        if self.pipeline is None or len(self.pipeline.data.file_list) == 0:
            self.logger.error("Input Error: please load data first.")
            return None

        # get the path(s) from a standard file selection screen
        file_path, _ = QFileDialog.getOpenFileName(self, caption="Load Result", dir=self.save_dir,
                                                    filter="HDF5 Files (*.hdf5);;All Files (*)")

        # do nothing in case no file selected
        if not file_path:
            return None

        # load result into instance
        try:
            self.pipeline.result = Result.load(file_name=file_path, logger=self.logger)
        except KeyError as e:
            self.logger.error(f"Incompatible file: {e}")

        # clear and update the plots
        self.update_plots(new_plot=False)

        return None

    def save_result(self):
        """Save result to a file using a dialog window."""
        # check whether a result is present
        if self.pipeline is None or len(self.pipeline.data.file_list) == 0:
            self.logger.error("Input Error: please load data first.")
            return None

        suggested_path = os.path.join(self.save_dir, self.pipeline.data.target_id + '_result.hdf5')
        file_path, _ = QFileDialog.getSaveFileName(self, caption="Save Data", dir=suggested_path,
                                                   filter="HDF5 Files (*.hdf5);;All Files (*)")

        # do nothing in case no file selected
        if not file_path:
            return None

        self.pipeline.result.save(file_path)

        return None

    def handle_result_signal(self):
        """Handle the emitted result signal: update the GUI with the results."""
        # display sinusoid parameters in the table
        self.update_table(display_err=True)

        # Update the plot area with the results
        self.update_plots(new_plot=False)

        return None

    def perform_analysis(self, func_name, *args, **kwargs):
        """Perform analysis on the loaded data and display results."""
        # check whether data is loaded
        if self.pipeline is None or len(self.pipeline.data.file_list) == 0:
            self.logger.error("Input Error: please provide data files.")
            return None

        # start a new thread for the analysis
        self.pipeline_thread.start_function(func_name, *args, **kwargs)

        return None

    def stop_analysis(self):
        """Stop the analysis, if it is running."""
        if self.pipeline_thread is not None:
            self.pipeline_thread.stop()

        return None

    def click_periodogram(self, x, y, button):
        """Handle click events on the periodogram plot."""
        # Guard against empty data
        if self.pipeline is None:
            self.append_text(f"Plot clicked at coordinates: ({x}, {y})")
            return None

        # Left click
        if button == 1:
            self.pipeline_thread.start_function('extract_approx', x)

        # Right click
        if button == 3:
            self.pipeline_thread.start_function('remove_approx', x)

        return None

    def add_harmonic_series(self):
        """Let the user add a base harmonic frequency and add the harmonic series."""
        if self.pipeline is None:
            QMessageBox.warning(self, "Warning", "Load data first.")
            return None

        # get some parameters
        min_val = self.pipeline.ts_model.pd_f0
        max_val = self.pipeline.ts_model.pd_fn
        text = f"Please enter a number between {min_val:1.2f} and {max_val:1.2f}."

        # open the dialog
        dialog = InputDialog("Add base harmonic frequency", text, "Exact")

        if dialog.exec():
            value, checked = dialog.get_values()

            try:
                value = float(value)
            except ValueError:
                warn_message = "Invalid input: Unable to convert to a floating point number."
                QMessageBox.warning(self, "Warning", warn_message)

                return None

            # if we made it here, add the harmonics
            self.pipeline_thread.start_function('add_harmonic_series', value, exact=checked)

        return None

    def show_settings_dialog(self):
        """Show a 'settings' dialog with configuration for the application."""
        dialog = gui_config.SettingsDialog(config=config, parent=self)

        if dialog.exec():
            # Update any dependent components with new configuration values
            screen = QApplication.primaryScreen()
            screen_size = screen.availableSize()
            h_size = int(screen_size.width() * config.h_size_frac)  # some fraction of the screen width
            v_size = int(screen_size.height() * config.v_size_frac)  # some fraction of the screen height
            self.setGeometry(100, 50, h_size, v_size)

        return None

    def show_about_dialog(self):
        """Show an 'about' dialog with information about the application."""
        version = hlp.get_version()
        message = (f"STAR SHINE version {version}\n"
                   "Satellite Time-series Analysis Routine "
                   "using Sinusoids and Harmonics through Iterative Non-linear Extraction\n"
                   "Repository: https://github.com/LucIJspeert/star_shine\n"
                   "Code written by: Luc IJspeert")
        QMessageBox.about(self, "About", message)

        return None


class InputDialog(QDialog):
    def __init__(self, title, text1, text2):
        super().__init__()

        self.setWindowTitle(title)
        layout = QFormLayout()

        # Create a label
        label = QLabel(text1)

        # Create a QLineEdit for input with double validation
        self.line_edit = QLineEdit()
        layout.addRow(label, self.line_edit)

        # Create a label and checkbox for additional option
        checkbox_label = QLabel(text2)
        self.checkbox = QCheckBox()
        layout.addRow(checkbox_label, self.checkbox)

        # Create a button to accept input
        ok_button = QPushButton("Accept")
        ok_button.clicked.connect(self.accept)
        layout.addWidget(ok_button)

        self.setLayout(layout)

    def get_values(self):
        return self.line_edit.text(), self.checkbox.isChecked()


def launch_gui():
    """Launch the Star Shine GUI."""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    launch_gui()
