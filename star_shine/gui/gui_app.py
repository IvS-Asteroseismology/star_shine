"""STAR SHINE
Satellite Time-series Analysis Routine using Sinusoids and Harmonics through Iterative Non-linear Extraction

This Python module contains the graphical user interface.

Code written by: Luc IJspeert
"""
import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QMenuBar, QMessageBox
from PySide6.QtGui import QAction

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set some window things
        self.setWindowTitle("Star Shine")
        self.setGeometry(300, 200, 1600, 1000)  # x, y, width, height

        # Create the menu bar
        menu_bar = QMenuBar()
        self.setMenuBar(menu_bar)

        # Add "File" menu
        file_menu = menu_bar.addMenu("File")

        # Add actions to "File" menu
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)  # Connect the action to the close method
        file_menu.addAction(exit_action)

        # Add "Info" menu
        info_menu = menu_bar.addMenu("Info")

        # Add actions to "Info" menu
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about_dialog)  # Connect the action to a custom method
        info_menu.addAction(about_action)

    def show_about_dialog(self):
        QMessageBox.about(self, "About", "This is a simple PySide application.")


def launch_gui():
    """Launch the Star Shine GUI"""
    # the main application class with command line arguments
    app = QApplication(sys.argv)

    # open the main window
    window = MainWindow()
    window.show()

    # make sure it exits correctly
    sys.exit(app.exec())


if __name__ == "__main__":
    launch_gui()
