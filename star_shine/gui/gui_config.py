"""STAR SHINE
Satellite Time-series Analysis Routine using Sinusoids and Harmonics through Iterative Non-linear Extraction

This Python module contains the settings dialog for the graphical user interface.

Code written by: Luc IJspeert
"""
from PySide6.QtWidgets import QDialog, QHBoxLayout, QVBoxLayout, QLineEdit, QPushButton, QLabel, QFormLayout
from PySide6.QtWidgets import QMessageBox


class SettingsDialog(QDialog):
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self.setWindowTitle("Settings")

        layout = QVBoxLayout()
        form_layout = QFormLayout()

        # Create input fields for each setting
        self.data_dir_field = QLineEdit(self.config.data_dir)
        self.save_dir_field = QLineEdit(self.config.save_dir)
        self.h_size_frac_field = QLineEdit(str(self.config.h_size_frac))
        self.v_size_frac_field = QLineEdit(str(self.config.v_size_frac))

        form_layout.addRow(QLabel("Data Directory:"), self.data_dir_field)
        form_layout.addRow(QLabel("Save Directory:"), self.save_dir_field)
        form_layout.addRow(QLabel("Horizontal Size Fraction:"), self.h_size_frac_field)
        form_layout.addRow(QLabel("Vertical Size Fraction:"), self.v_size_frac_field)

        layout.addLayout(form_layout)

        button_box = QHBoxLayout()
        save_button = QPushButton("Save")
        cancel_button = QPushButton("Cancel")

        save_button.clicked.connect(self.save_settings)
        cancel_button.clicked.connect(self.reject)

        button_box.addWidget(save_button)
        button_box.addWidget(cancel_button)

        layout.addLayout(button_box)
        self.setLayout(layout)

    def save_settings(self):
        try:
            # Update the configuration with new values
            self.config.data_dir = self.data_dir_field.text()
            self.config.save_dir = self.save_dir_field.text()
            self.config.h_size_frac = float(self.h_size_frac_field.text())
            self.config.v_size_frac = float(self.v_size_frac_field.text())

            self.accept()  # Close the dialog
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Invalid input for size fractions. Please enter numbers.")