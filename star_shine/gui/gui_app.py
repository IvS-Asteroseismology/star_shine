"""STAR SHINE
Satellite Time-series Analysis Routine using Sinusoids and Harmonics through Iterative Non-linear Extraction

This Python module contains the graphical user interface.

Code written by: Luc IJspeert
"""
import sys

from PySide6.QtWidgets import QApplication, QMainWindow, QMenuBar, QMessageBox, QVBoxLayout, QWidget, QPushButton, \
    QTextEdit, QLineEdit, QLabel, QHBoxLayout
from PySide6.QtGui import QAction

import star_shine.config.helpers as hlp


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set some window things
        self.setWindowTitle("Star Shine")
        self.setGeometry(300, 200, 1600, 1000)  # x, y, width, height

        # Create the central widget and set the layout
        self._setup_central_widget()

        # Create the menu bar
        self._setup_menu_bar()

        # Create input fields and buttons
        self._create_input_fields()
        self._create_buttons()
        self._create_output_area()

    def _setup_central_widget(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

    def _setup_menu_bar(self):
        menu_bar = QMenuBar()
        self.setMenuBar(menu_bar)

        # Add "File" menu
        file_menu = menu_bar.addMenu("File")
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)  # Connect the action to the close method
        file_menu.addAction(exit_action)

        # Add "Info" menu
        info_menu = menu_bar.addMenu("Info")
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about_dialog)  # Connect the action to a custom method
        info_menu.addAction(about_action)

    def _create_input_fields(self):
        # Input field for file path
        input_layout = QHBoxLayout()
        self.file_path_label = QLabel("File Path:")
        self.file_path_edit = QLineEdit()
        input_layout.addWidget(self.file_path_label)
        input_layout.addWidget(self.file_path_edit)

        # Add layout to main layout
        self.layout.addLayout(input_layout)

    def _create_buttons(self):
        # Button for starting analysis
        button_layout = QHBoxLayout()
        self.analyze_button = QPushButton("Analyze")
        self.analyze_button.clicked.connect(self.perform_analysis)  # Connect the action to a custom method
        button_layout.addWidget(self.analyze_button)

        # Add layout to main layout
        self.layout.addLayout(button_layout)

    def _create_output_area(self):
        # Text area for displaying results
        self.result_label = QLabel("Results:")
        self.result_text_edit = QTextEdit()
        self.result_text_edit.setReadOnly(True)  # Make the text edit read-only

        # Add widgets to main layout
        self.layout.addWidget(self.result_label)
        self.layout.addWidget(self.result_text_edit)

    def perform_analysis(self):
        file_path = self.file_path_edit.text().strip()
        if not file_path:
            QMessageBox.warning(self, "Input Error", "Please provide a valid file path.")
            return

        # Dummy analysis function (replace with actual analysis logic)
        results = f"Analysis completed for {file_path}. Results:\n\n" \
                  f"- Sinusoidal Component: 0.85\n- Harmonic Component: 0.15"

        self.result_text_edit.setPlainText(results)

    def show_about_dialog(self):
        version = hlp.get_version()
        message = (f"STAR SHINE version {version}\n"
                   "Satellite Time-series Analysis Routine "
                   "using Sinusoids and Harmonics through Iterative Non-linear Extraction\n"
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
