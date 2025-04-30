"""STAR SHINE
Satellite Time-series Analysis Routine using Sinusoids and Harmonics through Iterative Non-linear Extraction

This Python module contains the graphical user interface.

Code written by: Luc IJspeert
"""
import os
import sys

from PySide6.QtWidgets import QApplication, QWidget, QMainWindow, QVBoxLayout, QHBoxLayout, QSplitter, QMenuBar
from PySide6.QtWidgets import QLabel, QTextEdit, QLineEdit, QFileDialog, QMessageBox, QPushButton
from PySide6.QtWidgets import QTableView, QHeaderView
from PySide6.QtGui import QAction, QFont, QScreen, QStandardItemModel, QStandardItem, QTextCursor
from PySide6.QtCore import Signal

from star_shine.core import utility as ut
from star_shine.api import Data, Result, Pipeline
from star_shine.gui import gui_log, gui_plot, gui_analysis
from star_shine.config import helpers as hlp


# load configuration
config = hlp.get_config()


class MainWindow(QMainWindow):
    """The main window of the Star Shine application.

    Contains a graphical user interface for loading data, performing analysis,
    displaying results, and visualizing plots.
    """
    # setup signal receiving
    log_signal = Signal(str)
    result_signal = Signal(object)

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

        # Connect signals to slots
        self.log_signal.connect(self.append_text)
        self.result_signal.connect(self.receive_results)

        # custom gui-specific logger
        self.logger = gui_log.get_custom_gui_logger(self.log_signal, 'gui_logger', '')

        # add the api classes for functionality
        self.data_instance = Data()
        self.pipeline_instance = None
        self.pipeline_thread = None
        self.result_instance = Result()

        # some things that are needed
        self.data_dir = config.data_dir
        self.save_dir = config.save_dir
        self.save_subdir = ''

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

    def _setup_menu_bar(self):
        """Set up the menu bar with file and info menus."""
        menu_bar = QMenuBar()
        self.setMenuBar(menu_bar)

        # Add "File" menu and set it up
        file_menu = menu_bar.addMenu("File")
        self._setup_file_menu(file_menu)

        # Add "Info" menu
        info_menu = menu_bar.addMenu("Info")
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about_dialog)
        info_menu.addAction(about_action)

    def _setup_file_menu(self, file_menu):
        """Set up the file menu."""
        # Add "Load Data" button to "File" menu
        load_data_action = QAction("Load Data", self)
        load_data_action.triggered.connect(self.load_data_external)
        file_menu.addAction(load_data_action)

        # Add "Save Location" button to "File" menu
        set_save_location_action = QAction("Save Location", self)
        set_save_location_action.triggered.connect(self.set_save_location)
        file_menu.addAction(set_save_location_action)

        # Add a horizontal separator
        file_menu.addSeparator()

        # Add "Load Data Object" button to "File" menu
        load_data_object_action = QAction("Load Data Object", self)
        load_data_object_action.triggered.connect(self.load_data)
        file_menu.addAction(load_data_object_action)

        # Add "Save Data Object" button to "File" menu
        save_data_object_action = QAction("Save Data Object", self)
        save_data_object_action.triggered.connect(self.save_data)
        file_menu.addAction(save_data_object_action)

        # Add "Load Result Object" button to "File" menu
        load_result_object_action = QAction("Load Result Object", self)
        load_result_object_action.triggered.connect(self.load_result)
        file_menu.addAction(load_result_object_action)

        # Add "Save Result Object" button to "File" menu
        save_result_object_action = QAction("Save Result Object", self)
        save_result_object_action.triggered.connect(self.save_result)
        file_menu.addAction(save_result_object_action)

        # Add a horizontal separator
        file_menu.addSeparator()

        # Add "Exit" button to "File" menu
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

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

        # Input field for file path
        input_sub_layout = QHBoxLayout()
        self.file_path_label = QLabel("File Path:")
        self.file_path_edit = QLineEdit()
        input_sub_layout.addWidget(self.file_path_label)
        input_sub_layout.addWidget(self.file_path_edit)
        l_col_layout.addLayout(input_sub_layout)

        # Button for starting analysis
        self.analyze_button = QPushButton("Analyze")
        self.analyze_button.clicked.connect(self.perform_analysis)  # Connect the action to a custom method
        l_col_layout.addWidget(self.analyze_button)

        # Log area
        self.text_field = QTextEdit()
        self.text_field.setReadOnly(True)  # Make the text edit read-only
        self.text_field.setPlainText("Log\n")
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

        # lower plot area for the periodogram
        self.lower_plot_area = gui_plot.PlotWidget(title='Periodogram', xlabel='frequency', ylabel='amplitude')
        r_col_layout.addWidget(self.lower_plot_area)

        return r_col_widget

    def append_text(self, text):
        """Append a line of text at the end of the plain text output box.

        Parameters
        ----------
        text: str
            The text to append.

        Returns
        -------
        None
        """
        cursor = self.text_field.textCursor()
        cursor.movePosition(QTextCursor.End)  # Move cursor to the end of the text
        cursor.insertText(text + '\n')  # insert the text
        self.text_field.setTextCursor(cursor)
        self.text_field.ensureCursorVisible()

        return None

    def update_table(self, display_errors=True):
        """Fill the table with the given data.

        Returns
        -------
        None
        """
        # get the result parameters
        col1 = self.result_instance.f_n
        col2 = self.result_instance.a_n
        col3 = self.result_instance.ph_n
        col1_err = self.result_instance.f_n_err
        col2_err = self.result_instance.a_n_err
        col3_err = self.result_instance.ph_n_err

        # display sinusoid parameters in the table
        self.table_model.setRowCount(len(col1))
        for row, row_items in enumerate(zip(col1, col2, col3, col1_err, col2_err, col3_err)):
            # convert to strings
            c1 = ut.float_to_str_scientific(row_items[0], row_items[3], error=display_errors, brackets=False)
            c2 = ut.float_to_str_scientific(row_items[1], row_items[4], error=display_errors, brackets=False)
            c3 = ut.float_to_str_scientific(row_items[2], row_items[5], error=display_errors, brackets=False)

            # convert to table items
            c1_item = QStandardItem(c1)
            c2_item = QStandardItem(c2)
            c3_item = QStandardItem(c3)

            # insert into table
            self.table_model.setItem(row, 0, c1_item)
            self.table_model.setItem(row, 1, c2_item)
            self.table_model.setItem(row, 2, c3_item)

        return None

    def update_plots(self):
        """Update the plotting area with the current data."""
        # clear the plots
        self.upper_plot_area.clear_plot()
        self.lower_plot_area.clear_plot()

        # update the plots with data
        time, flux = self.data_instance.time, self.data_instance.flux
        freqs, ampls = self.data_instance.periodogram()
        self.upper_plot_area.scatter(time, flux)
        self.lower_plot_area.plot(freqs, ampls)

        # include result attributes if present
        if self.result_instance.target_id != '':
            model = self.pipeline_instance.model_linear()
            model += self.pipeline_instance.model_sinusoid()
            freqs, ampls = self.pipeline_instance.periodogram(residual=True)
            self.upper_plot_area.plot(time, model)
            self.lower_plot_area.plot(freqs, ampls)

        return None

    def set_save_location(self):
        """Open a dialog to select the save location."""
        # Open a directory selection dialog
        new_dir = QFileDialog.getExistingDirectory(self, caption="Select Save Location", dir=self.save_dir)

        if new_dir:
            self.save_dir = new_dir
            self.append_text(f"Save location set to: {self.save_dir}")

        return None

    def load_data_external(self):
        """Read data from a file or multiple files using a dialog window."""
        # get the path(s) from a standard file selection screen
        file_paths, _ = QFileDialog.getOpenFileNames(self, caption="Read Data", dir=self.save_dir,
                                                     filter="All Files (*)")

        # do nothing in case no file(s) selected
        if not file_paths:
            return None

        # load data into instance
        self.data_instance = Data.load_data(file_list=file_paths, data_dir='', target_id='', data_id='',
                                            logger=self.logger)
        self.save_subdir = f"{self.data_instance.target_id}_analysis"

        # clear and update the plots
        self.update_plots()

        return None

    def load_data(self):
        """Load data from a file using a dialog window."""
        # get the path(s) from a standard file selection screen
        file_path, _ = QFileDialog.getOpenFileName(self, caption="Load Data", dir=self.save_dir,
                                                    filter="HDF5 Files (*.hdf5);;All Files (*)")

        # do nothing in case no file selected
        if not file_path:
            return None

        # load data into instance
        self.data_instance = Data.load(file_name=file_path, data_dir='', logger=self.logger)
        self.save_subdir = f"{self.data_instance.target_id}_analysis"

        # clear and update the plots
        self.update_plots()

        return None

    def save_data(self):
        """Save data to a file using a dialog window."""
        suggested_path = os.path.join(self.save_dir, self.data_instance.target_id + '_data.hdf5')
        file_path, _ = QFileDialog.getSaveFileName(self, caption="Save Data", dir=suggested_path,
                                                   filter="HDF5 Files (*.hdf5);;All Files (*)")

        # do nothing in case no file selected
        if not file_path:
            return None

        self.data_instance.save(file_path)

        return None

    def load_result(self):
        """Load result from a file using a dialog window."""
        # get the path(s) from a standard file selection screen
        file_path, _ = QFileDialog.getOpenFileName(self, caption="Load Result", dir=self.save_dir,
                                                    filter="HDF5 Files (*.hdf5);;All Files (*)")

        # do nothing in case no file selected
        if not file_path:
            return None

        # load result into instance
        self.result_instance = Result.load(file_name=file_path, logger=self.logger)

        # clear and update the plots
        self.update_plots()

        return None

    def save_result(self):
        """Save result to a file using a dialog window."""
        suggested_path = os.path.join(self.save_dir, self.data_instance.target_id + '_result.hdf5')
        file_path, _ = QFileDialog.getSaveFileName(self, caption="Save Data", dir=suggested_path,
                                                   filter="HDF5 Files (*.hdf5);;All Files (*)")

        # do nothing in case no file selected
        if not file_path:
            return None

        self.result_instance.save(file_path)

        return None

    def receive_results(self, result):
        """Handle the results emitted from the analysis thread."""
        # Update the GUI with the results
        if result is None:
            return None

        # set result instance to received result
        self.result_instance = result

        # display sinusoid parameters in the table
        self.update_table(display_errors=True)

        # Update the plot area with the results
        self.update_plots()

        return None

    def perform_analysis(self):
        """Perform analysis on the loaded data and display results."""
        # check whether data is loaded
        if len(self.data_instance.file_list) == 0:
            QMessageBox.warning(self, "Input Error", "Please provide data files.")
            return None

        # for saving, make a folder if not there yet
        full_dir = os.path.join(self.save_dir, self.save_subdir)
        if not os.path.isdir(full_dir):
            os.mkdir(full_dir)  # create the subdir

        # redirect logging to text output
        logger = gui_log.get_custom_gui_logger(self.log_signal, self.data_instance.target_id, full_dir)

        # Perform analysis using your Pipeline class
        self.pipeline_instance = Pipeline(data=self.data_instance, save_dir=self.save_dir, logger=logger)

        # set up and start a new thread for the analysis
        self.pipeline_thread = gui_analysis.PipelineThread(self.pipeline_instance, self.result_signal)
        self.pipeline_thread.start()

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


def launch_gui():
    """Launch the Star Shine GUI."""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    launch_gui()
