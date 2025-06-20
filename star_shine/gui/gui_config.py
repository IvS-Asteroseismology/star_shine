"""STAR SHINE
Satellite Time-series Analysis Routine using Sinusoids and Harmonics through Iterative Non-linear Extraction

This Python module contains the settings dialog for the graphical user interface.

Code written by: Luc IJspeert
"""
from PySide6.QtWidgets import QDialog, QHBoxLayout, QVBoxLayout, QFormLayout, QLineEdit, QPushButton, QLabel, QCheckBox
from PySide6.QtWidgets import QMessageBox, QFrame

from star_shine.api.main import update_config, save_config
from star_shine.config.helpers import get_config


# load configuration
config = get_config()


class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.config = config
        self.setWindowTitle("Settings")

        layout = QHBoxLayout()
        left_layout = QVBoxLayout()

        # create the data and file settings form
        left_layout.addWidget(QLabel("Data and File Settings"))
        data_file_form_layout = self._create_data_file_settings()
        left_layout.addLayout(data_file_form_layout)

        # Create a horizontal line as a divider
        h_line = QFrame()
        h_line.setFrameShape(QFrame.HLine)
        h_line.setFrameShadow(QFrame.Sunken)
        left_layout.addWidget(h_line)

        # create the tabulated file settings form
        left_layout.addWidget(QLabel("Tabulated File Settings"))
        data_file_form_layout = self._create_tabulated_settings()
        left_layout.addLayout(data_file_form_layout)

        # Create a horizontal line as a divider
        h_line = QFrame()
        h_line.setFrameShape(QFrame.HLine)
        h_line.setFrameShadow(QFrame.Sunken)
        left_layout.addWidget(h_line)

        # create the fits file settings form
        left_layout.addWidget(QLabel("Fits File Settings"))
        data_file_form_layout = self._create_fits_settings()
        left_layout.addLayout(data_file_form_layout)

        layout.addLayout(left_layout)

        # Create a vertical line as a divider
        v_line = QFrame()
        v_line.setFrameShape(QFrame.VLine)
        v_line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(v_line)

        # create the GUI settings on the right side
        right_layout = QVBoxLayout()

        # create the extraction settings form
        right_layout.addWidget(QLabel("Extraction Settings"))
        extraction_form_layout = self._create_extraction_settings()
        right_layout.addLayout(extraction_form_layout)

        # Create a horizontal line as a divider
        h_line = QFrame()
        h_line.setFrameShape(QFrame.HLine)
        h_line.setFrameShadow(QFrame.Sunken)
        right_layout.addWidget(h_line)

        # create the GUI settings form
        right_layout.addWidget(QLabel("GUI Settings"))
        gui_form_layout = self._create_gui_settings()
        right_layout.addLayout(gui_form_layout)

        # the save/cancel buttons
        button_box = QHBoxLayout()
        apply_button = QPushButton("Apply")
        apply_button.clicked.connect(self.apply_settings)
        button_box.addWidget(apply_button)

        save_button = QPushButton("Save")
        save_button.clicked.connect(self.save_settings)
        button_box.addWidget(save_button)

        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_box.addWidget(cancel_button)

        right_layout.addLayout(button_box)

        layout.addLayout(right_layout)

        self.setLayout(layout)

    def _create_extraction_settings(self):
        """Create the settings form for the extraction settings."""
        form_layout = QFormLayout()

        # Create input fields for each setting
        self.select_next_field = QLineEdit(self.config.select_next)
        form_layout.addRow(QLabel("Select Next Sinusoid:"), self.select_next_field)

        self.optimise_step_field = QCheckBox()
        self.optimise_step_field.setChecked(self.config.optimise_step)
        form_layout.addRow(QLabel("Optimise Every Step:"), self.optimise_step_field)

        self.replace_step_field = QCheckBox()
        self.replace_step_field.setChecked(self.config.replace_step)
        form_layout.addRow(QLabel("Replace Every Step:"), self.replace_step_field)

        self.bic_thr_field = QLineEdit(str(self.config.bic_thr))
        form_layout.addRow(QLabel("BIC Threshold:"), self.bic_thr_field)

        self.snr_thr_field = QLineEdit(str(self.config.snr_thr))
        form_layout.addRow(QLabel("SNR Threshold:"), self.snr_thr_field)

        self.nyquist_factor_field = QLineEdit(str(self.config.nyquist_factor))
        form_layout.addRow(QLabel("Nyquist Factor:"), self.nyquist_factor_field)

        self.resolution_factor_field = QLineEdit(str(self.config.resolution_factor))
        form_layout.addRow(QLabel("Resolution Factor:"), self.resolution_factor_field)

        self.window_width_field = QLineEdit(str(self.config.window_width))
        form_layout.addRow(QLabel("Window Width:"), self.window_width_field)

        return form_layout

    def _create_data_file_settings(self):
        """Create the settings form for the data and file settings."""
        form_layout = QFormLayout()

        self.overwrite_field = QCheckBox()
        self.overwrite_field.setChecked(self.config.overwrite)
        form_layout.addRow(QLabel("Overwrite files:"), self.overwrite_field)

        # data_dir is not used in the gui for simplicity

        self.save_dir_field = QLineEdit(self.config.save_dir)
        form_layout.addRow(QLabel("Save Directory:"), self.save_dir_field)

        self.save_ascii_field = QCheckBox()
        self.save_ascii_field.setChecked(self.config.save_ascii)
        form_layout.addRow(QLabel("Save ASCII files:"), self.save_ascii_field)

        return form_layout

    def _create_tabulated_settings(self):
        """Create the settings form for the tabulated file settings."""
        form_layout = QFormLayout()

        self.cn_time_field = QLineEdit(self.config.cn_time)
        form_layout.addRow(QLabel("Column Name For time:"), self.cn_time_field)

        self.cn_flux_field = QLineEdit(self.config.cn_flux)
        form_layout.addRow(QLabel("Column Name For flux:"), self.cn_flux_field)

        self.cn_flux_err_field = QLineEdit(self.config.cn_flux_err)
        form_layout.addRow(QLabel("Column Name For flux_err:"), self.cn_flux_err_field)

        return form_layout

    def _create_fits_settings(self):
        """Create the settings form for the FITS file settings."""
        form_layout = QFormLayout()

        self.cf_time_field = QLineEdit(self.config.cf_time)
        form_layout.addRow(QLabel("Column Name For time:"), self.cf_time_field)

        self.cf_flux_field = QLineEdit(self.config.cf_flux)
        form_layout.addRow(QLabel("Column Name For flux:"), self.cf_flux_field)

        self.cf_flux_err_field = QLineEdit(self.config.cf_flux_err)
        form_layout.addRow(QLabel("Column Name For flux_err:"), self.cf_flux_err_field)

        self.cf_quality_field = QLineEdit(self.config.cf_quality)
        form_layout.addRow(QLabel("Column Name For quality:"), self.cf_quality_field)

        self.apply_q_flags_field = QCheckBox()
        self.apply_q_flags_field.setChecked(self.config.apply_q_flags)
        form_layout.addRow(QLabel("Apply Quality Flags:"), self.apply_q_flags_field)

        self.halve_chunks_field = QCheckBox()
        self.halve_chunks_field.setChecked(self.config.halve_chunks)
        form_layout.addRow(QLabel("Halve Time Chunks:"), self.halve_chunks_field)

        return form_layout

    def _create_gui_settings(self):
        """Create the settings form for the GUI settings."""
        form_layout = QFormLayout()

        self.dark_mode_field = QCheckBox()
        self.dark_mode_field.setChecked(False)
        form_layout.addRow(QLabel("Dark Mode [WIP]:"), self.dark_mode_field)

        self.h_size_frac_field = QLineEdit(str(self.config.h_size_frac))
        form_layout.addRow(QLabel("Horizontal Size Fraction:"), self.h_size_frac_field)

        self.v_size_frac_field = QLineEdit(str(self.config.v_size_frac))
        form_layout.addRow(QLabel("Vertical Size Fraction:"), self.v_size_frac_field)

        return form_layout

    def apply_settings(self):
        """Apply the settings to the configuration"""
        try:
            setting_dict = {}

            # make a settings dictionary
            setting_dict['overwrite'] = self.overwrite_field.isChecked()
            setting_dict['save_dir'] = self.save_dir_field.text()
            setting_dict['save_ascii'] = self.save_ascii_field.isChecked()

            setting_dict['cn_time'] = self.cn_time_field.text()
            setting_dict['cn_flux'] = self.cn_flux_field.text()
            setting_dict['cn_flux_err'] = self.cn_flux_err_field.text()

            setting_dict['cf_time'] = self.cf_time_field.text()
            setting_dict['cf_flux'] = self.cf_flux_field.text()
            setting_dict['cf_flux_err'] = self.cf_flux_err_field.text()
            setting_dict['cf_quality'] = self.cf_quality_field.text()
            setting_dict['apply_q_flags'] = self.apply_q_flags_field.isChecked()
            setting_dict['halve_chunks'] = self.halve_chunks_field.isChecked()

            setting_dict['select_next'] = self.select_next_field.text()
            setting_dict['optimise_step'] = self.optimise_step_field.isChecked()
            setting_dict['replace_step'] = self.replace_step_field.isChecked()
            setting_dict['bic_thr'] = float(self.bic_thr_field.text())
            setting_dict['snr_thr'] = float(self.snr_thr_field.text())
            setting_dict['nyquist_factor'] = float(self.nyquist_factor_field.text())
            setting_dict['resolution_factor'] = float(self.resolution_factor_field.text())
            setting_dict['window_width'] = float(self.window_width_field.text())

            setting_dict['dark_mode'] = self.dark_mode_field.isChecked()
            setting_dict['h_size_frac'] = float(self.h_size_frac_field.text())
            setting_dict['v_size_frac'] = float(self.v_size_frac_field.text())

            # Update the configuration with new values
            update_config(settings=setting_dict)
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Invalid input for settings.")

        # Close the dialog
        self.accept()

    def save_settings(self):
        """Save the settings form to disk."""
        self.apply_settings()

        try:
            # Save the configuration to a file
            save_config()
        except ValueError:
            QMessageBox.warning(self, "IO Error", "Error while saving config.")
