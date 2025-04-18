"""STAR SHINE
Satellite Time-series Analysis Routine using Sinusoids and Harmonics through Iterative Non-linear Extraction

This Python module contains the graphical user interface.

Code written by: Luc IJspeert
"""
import sys

from PySide6.QtWidgets import QApplication, QWidget, QMainWindow, QVBoxLayout, QHBoxLayout, QSplitter, QMenuBar
from PySide6.QtWidgets import QLabel, QPushButton, QTextEdit, QLineEdit, QFileDialog, QMessageBox, QTableView
from PySide6.QtGui import QAction, QFont, QScreen, QStandardItemModel

import matplotlib as mpl
import matplotlib.figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from star_shine.api import Data, Result, Pipeline
from star_shine.config import helpers as hlp


# load configuration
config = hlp.get_config()


class PlotWidget(QWidget):
    def __init__(self, title="Plot"):
        super().__init__()
        self.figure = mpl.figure.Figure()
        self.canvas = FigureCanvas(self.figure)
        self.title = title
        self.ax = self.figure.add_subplot(111)
        # self.ax.set_title(title)

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def plot(self, x, y):
        self.ax.clear()
        self.ax.plot(x, y)
        self.ax.set_title(self.title)
        self.canvas.draw()


class MainWindow(QMainWindow):
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
        self.setFont(font)

        # Create the central widget and set the layout
        self._setup_central_widget()

        # Create the menu bar
        self._setup_menu_bar()

        # Add widgets to the layout
        self._add_widgets_to_layout()

        # add the api classes for functionality
        self.data_instance = Data()

    def _setup_central_widget(self):
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
        self.splitter.setSizes([h_size*2//7, h_size*2//7, h_size*3//7])

    def _setup_menu_bar(self):
        menu_bar = QMenuBar()
        self.setMenuBar(menu_bar)

        # Add "File" menu
        file_menu = menu_bar.addMenu("File")

        # Add "Load Data" button to "File" menu
        load_data_action = QAction("Load Data", self)
        load_data_action.triggered.connect(self.load_data)  # Connect the action to a custom method
        file_menu.addAction(load_data_action)

        # Add "Exit" button to "File" menu
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)  # Connect the action to the close method
        file_menu.addAction(exit_action)

        # Add "Info" menu
        info_menu = menu_bar.addMenu("Info")
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about_dialog)  # Connect the action to a custom method
        info_menu.addAction(about_action)

    def _create_left_column(self):
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

        # Text area for displaying results
        self.result_text_edit = QTextEdit()
        self.result_text_edit.setReadOnly(True)  # Make the text edit read-only
        self.result_text_edit.setPlainText("Output")
        l_col_layout.addWidget(self.result_text_edit)

        return l_col_widget

    def _create_middle_column(self):
        # create a vertical layout for in the middle column of the main layout
        m_col_widget = QWidget()
        m_col_layout = QVBoxLayout(m_col_widget)

        # Create the table view and model
        self.table_view = QTableView()
        self.table_model = QStandardItemModel(0, 3)  # Start with 0 rows and 3 columns
        self.table_model.setHorizontalHeaderLabels(["Frequency", "Amplitude", "Phase"])
        self.table_view.setModel(self.table_model)

        # Add the button and table view to the middle column layout
        m_col_layout.addWidget(self.table_view)

        return m_col_widget

    def _create_right_column(self):
        # create a vertical layout for in the right column of the main layout
        r_col_widget = QWidget()
        r_col_layout = QVBoxLayout(r_col_widget)

        # upper plot area for the data
        self.upper_plot_area = PlotWidget(title="Data")
        r_col_layout.addWidget(self.upper_plot_area)

        # lower plot area for the periodogram
        self.lower_plot_area = PlotWidget(title="Periodogram")
        r_col_layout.addWidget(self.lower_plot_area)

        return r_col_widget

    def load_data(self):
        # get the path(s) from a standard file selection screen
        file_paths, _ = QFileDialog.getOpenFileNames(self, caption="Load Data", dir="",
                                                     filter="All Files (*);;Text Files (*.txt)")

        # do nothing in case no file(s) selected
        if not file_paths:
            return None

        # load a data into instance
        print(file_paths)
        self.data_instance.load_data(file_paths)
        print(self.data_instance.time)

        # update the plots
        time, flux = self.data_instance.time, self.data_instance.flux
        freqs, ampls = self.data_instance.periodogram()
        self.upper_plot_area.plot(time, flux)
        self.lower_plot_area.plot(freqs, ampls)

        return None

    def perform_analysis(self):
        file_path = self.file_path_edit.text().strip()
        if not file_path:
            QMessageBox.warning(self, "Input Error", "Please provide a valid file path.")
            return

        # Load data using your Data class
        file_paths = [path.strip() for path in file_path.split(',')]
        data_instance = Data(file_paths)
        loaded_data = data_instance.load_data()

        if isinstance(loaded_data, dict) and 'x' in loaded_data and 'y' in loaded_data:
            x = np.array(loaded_data['x'])
            y = np.array(loaded_data['y'])

            # Perform analysis using your Pipeline class
            pipeline_instance = Pipeline(data=data_instance)
            analysis_results = pipeline_instance.run_analysis()

            # Format results using your Result class
            result_instance = Result(analysis_results=analysis_results)
            formatted_results = result_instance.format_results()

            self.result_text_edit.setPlainText(formatted_results)

            # Update plots if needed (already done in load_data)
        else:
            QMessageBox.warning(self, "Data Error", "Loaded data format is incorrect.")
            return

    def show_about_dialog(self):
        version = hlp.get_version()
        message = (f"STAR SHINE version {version}\n"
                   "Satellite Time-series Analysis Routine "
                   "using Sinusoids and Harmonics through Iterative Non-linear Extraction\n"
                   "Repository: https://github.com/LucIJspeert/star_shine\n"
                   "Code written by: Luc IJspeert")
        QMessageBox.about(self, "About", message)


def launch_gui():
    """Launch the Star Shine GUI"""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    launch_gui()
