"""
MIT License

Copyright (c) 2024 Jonathan Rabe

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""
##########################################################################
'''
This simple simulator is meant as an educational tool to help students understand
the working of a mosfet. Input files define the device sizing, which the simulator
loads and then visualizes. Calculations of currents, voltages, etc, are based on
simple second order mosfet model equations. Units are in microns for all axes.
A device file is needed titled device.json, which defines sizing and doping.
Jonathan C. Rabe, jonathan@nuvetech.org, 2023, linkedin.com/in/jonathanrabe
Buy me a coffee: https://www.buymeacoffee.com/jonathanrabe
References:
- Semiconductor Devices Physics and Technology by S. M. SZE
- Analog Integrated Circuit Design by David A. Johns & Ken Martin
'''
##########################################################################
'''
ToDo list:
- add a settings page and json file
- Offer user option to use a slider to change opacity in settings page
- Offer user option to control Vbs and update math for it
- Visualise in text which part is source, drain, channel, gate, etc
- Set text of colors based on device.json in: 'self.colorLabel.setText()'
- Can we make the 3D graph clickable to set the DC operating point?
Done:
+ Add 'buy me coffee' link on about page, as well as here in code at top
+ Can we visualize channel and pinch-off happening/forming?
+ Calculate fields in function 'update_VoltageandField_Values(self):'
+ Print device operating region and other parameters such as current
+ Offer user sliders and text input to control Vds and Vgs
+ Calculate number of charge carriers and set balls equal to ratio
+ Change particles to be visible through other device parts
+ Add visualize button for field
+ Add field color label
+ Draw the electric field visualization in 'if self.show_electric_field:'
+ Calculate by hand the device parameters and DC point, and compare with sim
+ Set device colors based on device.json file
+ Can we show a live IV characteristic curve which shows the operating point as a dot on the curve?
'''
##########################################################################


#Importing all the libraries we will need
import sys
import random
import json
import math
import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QSplitter,
    QMainWindow,
    QOpenGLWidget,
    QPushButton,
    QLabel,
    QSlider,
    QLineEdit,
    QDialog,
    QWidget,
    QScrollArea,
    QVBoxLayout,
    QVBoxLayout,
    QHBoxLayout,
    QGraphicsView,
    QGraphicsScene,
    QMessageBox
    )
from PyQt5.QtGui import QColor
from OpenGL.GL import *
from OpenGL.GLU import gluPerspective
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

##########################################################################
#Global variables:
###############################
#Settings:
#ToDo: change these to a json file with a settings window to change settings
#Device starting coordinates offsets (these move the entire loading process by the amounts below)
x_offset = -1.0
y_offset = 0.2
z_offset = 0.0
#To preserve and limit memory, change this to limit how many can be generated:
MaxRenderedCarriers = 250
# Known current range that can be visualized
min_current_uA = 0   # 0uA
max_current_uA = 125  # simulator locks values below this
# Min and max range of Vgs and Vds in volts:
min_vds = 0
max_vds = 4
min_vgs = 0
max_vgs = 6
numerical_Differentiation_Resolution = 1000 #In how many pieces should the ranges above be broken for numerical stuff
ChargeScalingFactor = 1e6*max_current_uA/40 #Every this many charges would be represented with a single ball
# Below is the timestep resolution (used in a timer)
TimeStep_ms = 42 #About 24 fps
# Resolution of the x,y,z electric field finite elements:
FieldResolutionX = 0.06 #In microns which is our standard unit for this sim
FieldResolutionY = 0.03 #In microns which is our standard unit for this sim
FieldResolutionZ = 0.4 #In microns which is our standard unit for this sim
# So that things don't happen super fast, we reduce the field:
Reduce_Electric_Field_Factor = 1.5e7
# Draw Field for only one slice through cross section (1=enable one slice, 0=draw them all)
DrawFieldOnlyOnce = 1
# Define the scale factor for the electric field vectors
field_Vector_Draw_scale = 0.9
# By how much should the channel width be exaggerated in the visualization?
ExaggerateChannel = 1.6 #A factor for visual purposes only
# Operating point values:
#ToDo: allow user to set temperture while operating
DeviceTemperature = 300 # Temperature in Kelvin
# Define custom colormap with three colors for different regions (cutoff, linear, sat)
custom_cmap_DC_Plot = mcolors.ListedColormap(['red', 'blue', 'yellow'])
#End of settings
###############################

###############################
#Globals:
# Physical universal constants (starts with uc_ which is short for Universal Constant):
uc_k = 1.380649e-23  # Boltzmann constant  
uc_q = 1.602176634e-19  # Charge of an electron in coulombs
uc_EpsFS = 8.854e-12 # Freespace permittivity
uc_ChargePerAmpereSecond = 6.241509074*1e18 # 1 Ampere = this many charges per second
Silicon_ni = 1.5e10  # Intrinsic carrier concentration in cm^-3 (assuming silicon)
EpsSi = 11.68 # Silicon relative permitivity
EpsSiOx = 3.9 # Silicon dioxide relative permitivity
# Charge carriers list [x, y, z coordinates]
charge_carriers = []
#End of globals
###############################
#Temporaries until we improve the model:
#ToDo: change below to a function that calculates real values
CarrierDeltaXperTimeStep = 0.04 #should be a function of at least mobility and electric field
#ChannelThickness = 0.2 #should be calculated #Remove later



##########################################################################################
class MosfetViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        # Create an instance of DeviceLoader and load device information
        self.device_loader = DeviceLoader()
        self.device_loader.load_device_info("device.json")  # Replace with the path to your device information file
        #####################################################
        #Physics parameters
        # Define the bounds for each dimension
        x_start = self.device_loader.physical_parameters.source_width + x_offset + self.device_loader.physical_parameters.min_x_source
        x_end =   self.device_loader.physical_parameters.L_SourcetoDrain_Microns + x_offset + self.device_loader.physical_parameters.max_x_source
        VisualChannelThickness = self.device_loader.physical_parameters.max_depletion_region_width_microns*ExaggerateChannel
        y_start = self.device_loader.physical_parameters.min_y_gate_oxide - VisualChannelThickness
        y_end = self.device_loader.physical_parameters.min_y_gate_oxide
        z_start = self.device_loader.physical_parameters.min_z_source
        z_end = self.device_loader.physical_parameters.max_z_source
        # Calculate the dimensions of the electric field matrix based on the specified resolution
        x_dim = int((x_end - x_start) / FieldResolutionX)
        y_dim = int((y_end - y_start) / FieldResolutionY)
        z_dim = int((z_end - z_start) / FieldResolutionZ)
        self.electric_field_matrix = np.empty((x_dim, y_dim, z_dim, 3))
        # Loop through the electric field matrix and populate it within the specified bounds
        for x in range(x_dim):
            for y in range(y_dim):
                for z in range(z_dim):
                    # Calculate the corresponding coordinates within the specified bounds
                    x_coord = x_start + x * FieldResolutionX
                    y_coord = y_start + y * FieldResolutionY
                    z_coord = z_start + z * FieldResolutionZ
                    # Check if the coordinates are within the specified bounds
                    if (
                        x_coord >= x_start and x_coord < x_end and
                        y_coord >= y_start and y_coord < y_end and
                        z_coord >= z_start and z_coord < z_end
                    ):
                        # Default values (1, 0, 0) for the electric field
                        electric_field_value = np.array([1, 0, 0])
                        # Store the value in the matrix
                        self.electric_field_matrix[x, y, z] = electric_field_value
                    else:
                        # Coordinates are outside the specified bounds, set to zero or any desired value
                        self.electric_field_matrix[x, y, z] = np.array([0, 1, 0]) #Visual flag for out of bounds
        # Create an instance of OpenGLWidget and pass device information
        self.glWidget = OpenGLWidget(self, self.device_loader.device_parts,
                                     self.device_loader.source_face,
                                     self.device_loader.drain_face,
                                     self.device_loader.gate_dielectric,
                                     self.device_loader.physical_parameters,
                                     self.electric_field_matrix)
        ######################################################################
        # User interface and GUI parameters
        # Create a button to reset the view to "Home"
        self.homeButton = QPushButton("Home", self)
        self.homeButton.clicked.connect(self.glWidget.reset_view)
        self.homeButton.setGeometry(10, 10, 100, 30)
        # Create a button to toggle axes visibility
        self.toggleAxesButton = QPushButton("Toggle Axes", self)
        self.toggleAxesButton.clicked.connect(self.glWidget.toggle_axes)
        self.toggleAxesButton.setGeometry(120, 10, 100, 30)
        # Create a button to toggle electric field visualization
        self.toggleElectricFieldButton = QPushButton("Toggle Field", self)
        self.toggleElectricFieldButton.clicked.connect(self.glWidget.toggle_electric_field)
        self.toggleElectricFieldButton.setGeometry(230, 10, 100, 30)
        # Create a button to display device parameters for a user
        self.displayParametersButton = QPushButton("Display Device Parameters", self)
        self.displayParametersButton.clicked.connect(self.display_parameters)
        self.displayParametersButton.setGeometry(10, 50, 210, 30)
        # Create a button to display credits
        self.displayCreditsButton = QPushButton("About", self)
        self.displayCreditsButton.clicked.connect(self.display_credits)
        self.displayCreditsButton.setGeometry(230, 50, 100, 30)
        # Create textfields for user input of voltages:
        self.vdsLineEdit = None
        self.vgsLineEdit = None
        # Create input fields for Vds and Vgs
        self.vdsLineEdit = QLineEdit(self)
        self.vdsLineEdit.setPlaceholderText(f"Vds ({min_vds}V - {max_vds}V)")
        self.vdsLineEdit.setGeometry(10, 50, 100, 30)
        self.vgsLineEdit = QLineEdit(self)
        self.vgsLineEdit.setPlaceholderText(f"Vgs ({min_vgs}V - {max_vgs}V)")
        self.vgsLineEdit.setGeometry(10, 90, 100, 30)
        # Connect the returnPressed signal to the update_VoltageandField_Values function
        self.vdsLineEdit.returnPressed.connect(self.update_VoltageandField_Values)
        self.vgsLineEdit.returnPressed.connect(self.update_VoltageandField_Values)
        # Create labels for the textfields
        self.vdsLabel = QLabel('Vds (V)', self)
        self.vdsLabel.setGeometry(120, 60, 30, 30)
        self.vdsLabel.setStyleSheet("QLabel { color : white; }")
        self.vgsLabel = QLabel('Vgs (V)', self)
        self.vgsLabel.setGeometry(120, 100, 30, 30)
        self.vgsLabel.setStyleSheet("QLabel { color : white; }")
        # Create slider for Vds
        self.vdsSlider = QSlider(self)
        self.vdsSlider.setOrientation(1)  # 1 means vertical orientation
        self.vdsSlider.setGeometry(160, 50, 30, 100)
        self.vdsSlider.setMinimum(0)
        self.vdsSlider.setMaximum(int(max_vds*1000))  # Assuming the range is from 0V to max_vds, multiplied by 1000 for integer representation
        # Create slider for Vgs
        self.vgsSlider = QSlider(self)
        self.vgsSlider.setOrientation(1)  # 1 means vertical orientation
        self.vgsSlider.setGeometry(200, 50, 30, 100)
        self.vgsSlider.setMinimum(0)
        self.vgsSlider.setMaximum(int(max_vgs*1000))  # Assuming the range is from 0V to max_vgs, multiplied by 1000 for integer representation
        # Connect sliders to update corresponding line edits
        self.vdsSlider.valueChanged.connect(self.update_vds_edit)
        self.vgsSlider.valueChanged.connect(self.update_vgs_edit)
        # Create a label to display current parameters (initially empty)
        self.paramLabel = QLabel(self)
        label_width = 200  # Adjust the label's width as needed
        label_margin = 10  # Adjust the margin as needed
        label_x = self.width() - label_width - label_margin
        label_y = 10  # Adjust the vertical position as needed
        self.paramLabel.setGeometry(label_x, label_y, label_width, 30)
        self.paramLabel.setStyleSheet("QLabel { color : white; }")
        # Create a label to display the color of the electric field:
        self.electricFieldLabel = QLabel(self)
        label_width = 200  # Adjust the label's width as needed
        label_margin = 10  # Adjust the margin as needed
        label_x = self.width() - label_width - label_margin
        label_y = 50  # Adjust the vertical position as needed
        self.electricFieldLabel.setGeometry(label_x, label_y, label_width, 30)
        self.electricFieldLabel.setStyleSheet("QLabel { color : white; }")
        # Create a label to display color codes
        self.colorLabel = QLabel(self)
        color_label_width = 200
        color_label_x = self.width() - color_label_width - label_margin
        color_label_y = 90  # Adjust the vertical position as needed
        self.colorLabel.setGeometry(color_label_x, color_label_y, color_label_width, 60)
        self.colorLabel.setStyleSheet("QLabel { color : white; }")
        # Set the initial text for the color code label
        # ToDo: Set the color codes based on your device information file
        self.colorLabel.setText("Color Codes:\nSource: Blue\nDrain: Green\nGate: Red")
        # Create a label to show region of operation and drain current:
        self.DC_OP_Label = QLabel(self)
        DC_OP_label_width = 210  # Adjust the width as needed
        DC_OP_label_x = self.width() - DC_OP_label_width - label_margin
        DC_OP_label_y = 90  # Adjust the vertical position as needed
        self.DC_OP_Label.setGeometry(DC_OP_label_x, DC_OP_label_y, DC_OP_label_width, 100)
        self.DC_OP_Label.setStyleSheet("QLabel { color : white; }")
        # Set the initial text for the DC_OP_Label label
        self.DC_OP_Label.setText("Region: Set Vds+Vgs\nIdrain: Demo 10uA\ngm: NA (Demo)\ngds: NA (Demo)\nVeff: TBD\ngm/Id: TBD")
        # Setup the plot for IV Characteristic:
        self.mpl_widget = IV_Curve_MatplotlibWidget(self.device_loader.physical_parameters, self)
        # Store a reference to the instance of IV_Curve_MatplotlibWidget
        self.iv_curve_widget_instance = self.mpl_widget 
        self.mpl_widget.setStyleSheet("background-color: rgba(255,255,255,0);border: 0px;")
        self.glWidget.setGeometry(0,0,1100,800)
        self.setCentralWidget(self.glWidget)
        # Set up the main window
        self.setWindowTitle("Interactive MOSFET Demo with device physics")
        self.setGeometry(100, 100, 1300, 900)
        ####################################################################################
        
        
    def update_vds_edit(self, value):
        self.vdsLineEdit.setText(str(value / 1000.0))  # Convert back to volts
        self.update_VoltageandField_Values()
    def update_vgs_edit(self, value):
        self.vgsLineEdit.setText(str(value / 1000.0))  # Convert back to volts
        self.update_VoltageandField_Values()
    def display_parameters(self):
        # Create and show the parameter display dialog
        param_dialog = ParameterDisplayDialog(self.device_loader.physical_parameters, self)
        param_dialog.exec_()    
    def display_credits(self):
        #Credits message
        #ToDo: add buyMeACoffee link into message
        # Customizable text for the pop-up box with a clickable link
        line1 = "MOSFET Viewer is a program that shows a user"
        line2 = "a 3D model of a device, with live characteristics"
        line3 = "calculated from a DC operating point"
        line4 = "Created by Jonathan Rabe"
        line5 = "jonathan@nuvetech.org"
        line6 = "2023"
        # Create a pop-up box with a QLabel for displaying rich text
        credits_box = QMessageBox(self)
        credits_box.setWindowTitle("About")
        label = QLabel(self)
        label.setOpenExternalLinks(True)  # Allow the link to be opened in a web browser
        label.setTextInteractionFlags(Qt.TextBrowserInteraction)
        label.setOpenExternalLinks(True)
        label.setTextFormat(Qt.RichText)
        label.setAlignment(Qt.AlignCenter)
        html_content = f"{line1}<br>{line2}<br>{line3}<br>{line4}<br>{line5}<br>{line6}<br><br>Visit our website: <a href='https://www.nuvetech.org'>www.nuvetech.org</a><br>Support the developer: <a href='https://www.buymeacoffee.com/jonathanrabe'>Buy me a coffee</a>"
        label.setText(html_content)
        credits_box.layout().addWidget(label)
        # Add a clickable link to the message
        credits_box.setDetailedText(f"{html_content}")
        credits_box.setStyleSheet("QLabel{min-width: 500px;min-height: 160px;}")
        credits_box.exec_()

        
    def update_lbl_txt_slide_positions(self):
        # This function will resize and update the positions of labels, textboxes, etc.
        # So that the GUI stays formatted as it should
        # Update label positions based on window size
        Y_lbl_offset = 10
        label_margin = 10  # Adjust the margin as needed
        label_height = 30  # Adjust the label height as needed Todo:calculate height based on \n's
        label_width = 210  # Adjust the label's width as needed
        label_x = self.width() - label_width - label_margin  # Calculate label X position
        label_y1 = label_margin + Y_lbl_offset  # Calculate label 1 Y position
        label_y2 = label_margin + label_height + 10 + Y_lbl_offset  # Calculate label 2 Y position
        label_y3 = label_margin + 2*label_height + 10 + Y_lbl_offset
        fieldWidths = self.vdsLineEdit.width() + 10
        self.paramLabel.setGeometry(label_x, label_y1, label_width, label_height)  # Set paramLabel position
        self.electricFieldLabel.setGeometry(label_x, label_y2, label_width, label_height) #Set field label
        self.colorLabel.setGeometry(label_x, label_y3, label_width, label_height*2)  # Set colorLabel position
        # Update input field positions
        self.vdsLineEdit.setGeometry(label_x, label_y3 + 2*label_height + 10, 100, 30)
        self.vgsLineEdit.setGeometry(label_x, label_y3 + 2*label_height + 90, 100, 30)
        # Update voltage labels and sliders positions:
        self.vdsLabel.setGeometry(label_x + fieldWidths, label_y3 + 2*label_height + 10, 100, 30)  # Set vdsLabel position
        self.vgsLabel.setGeometry(label_x + fieldWidths, label_y3 + 2*label_height + 90, 100, 30)  # Set vgsLabel position
        self.vdsSlider.setGeometry(label_x, label_y3 + 2*label_height + 50, 100, 30)
        self.vgsSlider.setGeometry(label_x, label_y3 + 2*label_height + 130, 100, 30)
        self.DC_OP_Label.setGeometry(label_x, label_y3 + 3*label_height + 140, 150, 100)
        self.mpl_widget.setGeometry(self.width()-500, self.height()-420, 500, 420)
    
    #ToDo: edit below to generate new electric fields
    def update_VoltageandField_Values(self):
        # When the user hits enter or changes the sliders, this function is called which reads the vgs and vds values from the text inputs
        # This function generates a 2D electric field based on these values - which is then used for drift of carriers
        # This whole function uses a local coordinate system, assuming index of 0,0,0 is coordinate 0,0,0. It is translated in rendering
        # section.
        try:
            # Get the values from the text fields
            input_vds_text = self.vdsLineEdit.text()
            input_vgs_text = self.vgsLineEdit.text()
            # Check for blank inputs - replace with 0
            if input_vds_text == '':
                input_vds_text = '0'
            if input_vgs_text == '':
                input_vgs_text = '0'
            # Attempt conversion
            vds_value = float(input_vds_text)
            vgs_value = float(input_vgs_text)
            # Clamp values to the specified range
            # ToDo: Tell user range here if they exceed it... for now, just clamp
            vds_value = max(min_vds, min(vds_value, max_vds))
            vgs_value = max(min_vgs, min(vgs_value, max_vgs))
            self.device_loader.physical_parameters.Vgs_V = vgs_value
            self.device_loader.physical_parameters.Vds_V = vds_value
            self.calculate_DC_operating_point() #Get the current, Veff, small signal parameters, etc
            # Get the dimensions of the electric field matrix
            x_dim, y_dim, z_dim, _ = self.electric_field_matrix.shape
            C_Length = self.device_loader.physical_parameters.L_SourcetoDrain_Microns/1e6
            C_Height = self.device_loader.physical_parameters.max_depletion_region_width_microns/1e6         
            Source_to_Drain_Field = vds_value / abs(C_Length)
            Body_to_Gate_Field = vgs_value / abs(C_Height)
            Total_Field = math.sqrt((Body_to_Gate_Field**2)+(Source_to_Drain_Field**2))
            pinch_off_index = x_dim #Start by assuming this, then we check
            X_Coord_Pinchoff = (x_dim*10)*FieldResolutionX # Starting assumption
            OperateRegion = self.device_loader.physical_parameters.OperatingRegion
            Vdsat = self.device_loader.physical_parameters.Vdsat_V
            Vth = self.device_loader.physical_parameters.threshold_voltage_V
            if OperateRegion == 1: # Then we are in linear. Let's generate a coordinate
                Vunder = (Vdsat - vds_value)/Vdsat  # How far are we below saturation as a fraction
                pinch_off_index = x_dim # Generate field pointing to pinchoff all the way to end
                X_Coord_Pinchoff = (x_dim*FieldResolutionX)*(1+(2*Vunder))
            elif OperateRegion == 2: # Then we are in saturation. We need to generate a coord and index
                Vover = abs(((max_vds - vds_value))/((max_vds-Vdsat))) #Just linearly interp for now with 10% headroom
                Vover = (Vover + 0.1)*0.90909 # Just a scaling so that pinchoff is never at x=0
                pinch_off_index = int(x_dim*Vover)
                X_Coord_Pinchoff = (pinch_off_index*FieldResolutionX)
            else:
                X_Coord_Pinchoff = (pinch_off_index*2)*FieldResolutionX
            # Populate the electric field matrix with calculated values
            for x_field_index in range(x_dim):
                for y_field_index in range(y_dim):
                    for z_field_index in range(z_dim):
                        if y_field_index == (y_dim - 1):  # Topmost layer - set to horizontal
                            electric_field_value = np.array([Total_Field/Reduce_Electric_Field_Factor, 0, 0])
                        elif x_field_index <= pinch_off_index:
                            # Compute pinchoff coordinate:
                            Coordinate_of_PinchOff = np.array([X_Coord_Pinchoff, y_dim*FieldResolutionY, z_field_index*FieldResolutionZ])
                            # Scale the field vector based on spatial resolution
                            Coordinate_of_Point_in_Field = np.array([x_field_index*FieldResolutionX, y_field_index*FieldResolutionY, z_field_index*FieldResolutionZ])
                            # Compute the vector from our coordinate to the pinchoff point
                            direction_vector = Coordinate_of_PinchOff - Coordinate_of_Point_in_Field
                            # Normalize this direction vector:
                            normalized_direction = direction_vector / np.linalg.norm(direction_vector)
                            # Multiply by length of vector to get field strength
                            electric_field_value = normalized_direction * (Total_Field/Reduce_Electric_Field_Factor)                           
                        else:
                            # Outside the channel region, electric field is small vertical
                            electric_field_value = np.array([0, Total_Field/Reduce_Electric_Field_Factor, 0])
                        # Store the value in the matrix
                        self.electric_field_matrix[x_field_index, y_field_index, z_field_index] = electric_field_value      
            # Trigger a refresh of the OpenGL widget to send it the new field matrix
            self.glWidget.update()
        except Exception as e:
            print(f"Error generating field: {e}")
            
    def calculate_DC_operating_point(self):
        # This function is meant to take the values from vgs, vds, etc, and call the required
        # computations so that we calculate the operating point. It also updates the UI to show
        # these values.
        _vgs = self.device_loader.physical_parameters.Vgs_V
        _vds = self.device_loader.physical_parameters.Vds_V
        _Idrain, _gmuS, region, _lambda, _gdsuS, _vdsat = Device_Mathematical_Model(_vgs, _vds, self.device_loader.physical_parameters)
        _vth = self.device_loader.physical_parameters.threshold_voltage_V
        self.device_loader.physical_parameters.Vdsat_V = _vdsat
        self.device_loader.physical_parameters.Idrain_uA = _Idrain
        self.device_loader.physical_parameters.OperatingRegion = region
        self.device_loader.physical_parameters.gm_uS = _gmuS
        self.device_loader.physical_parameters.lambda_PerV = _lambda
        self.device_loader.physical_parameters.gds_uS = _gdsuS
        # Adding more lines here: also do a ctrl+F for DC_OP_Label.setGeometry and increase height
        label_text = (
            f"Region: {region}\n"
            f"Idrain: {(_Idrain):.3f} uA\n"
            f"gm: {(_gmuS):.3f} uS\n"
            f"gds: {(_gdsuS):.3f} uS\n"
            f"Veff: {(_vgs - _vth):.3f} V"
        )
        if _Idrain != 0:
            label_text += f"\ngm/Id: {(_gmuS / _Idrain):.3f} /V"
        else:
            label_text += "\ngm/Id: undef"
        self.DC_OP_Label.setText(label_text)
        if self.iv_curve_widget_instance is not None:
            self.iv_curve_widget_instance.Refresh_DC_OP()
        
    
    def resizeEvent(self, event):
        # Handle window resize events
        super().resizeEvent(event)
        self.update_lbl_txt_slide_positions()
        

class OpenGLWidget(QOpenGLWidget):
    def __init__(self, parent, device_parts, source_face, drain_face, gate_dielectric, physical_parameters, electric_field_matrix):
        super().__init__(parent)
        self.source_face = source_face
        self.drain_face = drain_face
        self.gate_dielectric = gate_dielectric
        self.dev_physical_parameters = physical_parameters
        self.show_axes = False  # Flag to control axes visibility
        self.show_electric_field = False  # Initialize to False, don't show electric field by default
        self.rotationX = 0
        self.rotationY = 0
        self.lastPos = None
        self.show_second_cube = False
        self.zoom_factor = 1.0  # Initial zoom factor
        self.right_mouse_button_pressed = False
        self.last_right_mouse_pos = None
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.device_parts = device_parts  # Pass device_parts from MainWindow
        # Pass the electric_field_matrix
        self.electric_field_matrix = electric_field_matrix  
        # Create a timer for charge carrier animation
        self.charge_carrier_timer = QTimer(self)
        self.charge_carrier_timer.timeout.connect(self.update_charge_carriers)
        self.charge_carrier_timer.start(TimeStep_ms)  #timer to increment carrier positions
        # Set the physical parameters which come from the device loader class
        self.physical_parameters = physical_parameters
        

    def update_charge_carriers(self):
        # This function updates the position in the rendering of the charge carrier lumps,
        # and is called as often as is dictated by the timer charge_carrier_timer
        # This timer also uses the variable TimeStep_ms to dictate how many
        # milliseconds between rendering updates
        # ToDo: Use electric field to generate velocity vectors to move carriers, rather than a constant
        # We can access the field in: self.electric_field_matrix
        VisualChannelThickness = self.dev_physical_parameters.max_depletion_region_width_microns * ExaggerateChannel
        Terminate_x = self.dev_physical_parameters.L_SourcetoDrain_Microns + x_offset + self.dev_physical_parameters.max_x_source
        Spawn_Face_X = self.dev_physical_parameters.source_width + x_offset + self.dev_physical_parameters.min_x_source  # Fixed x position
        Spawn_z_min = self.dev_physical_parameters.min_z_source + z_offset
        Spawn_z_max = self.dev_physical_parameters.max_z_source + z_offset
        Spawn_y_min = self.dev_physical_parameters.min_y_gate_oxide - VisualChannelThickness + y_offset
        # Calculate the number of charge carriers to render based on the current
        current_to_render_uA = max(min(self.dev_physical_parameters.Idrain_uA, max_current_uA), min_current_uA)  # Ensure it's within the limits
        carriers_to_inject = ((current_to_render_uA / 3e12) * uc_ChargePerAmpereSecond * (TimeStep_ms / 1000) / ChargeScalingFactor)        
        # Get the dimensions of the electric field matrix
        x_dim_FieldMatrix, y_dim_FieldMatrix, z_dim_FieldMatrix, _ = self.electric_field_matrix.shape       
        # Update the positions of charge carriers based on electric field
        for carrier in charge_carriers:
            x, y, z = carrier
            # Shift the coordinates to match the positive values in the field matrix
            x_shifted = x - (Spawn_Face_X)
            y_shifted = y - (Spawn_y_min)
            z_shifted = z - (Spawn_z_min)
            # Get the electric field vector at the current carrier position
            X_index = int(x_shifted/FieldResolutionX)
            Y_index = int(y_shifted/FieldResolutionY)
            Z_index = int(z_shifted/FieldResolutionZ)
            # Keep Z index within limits
            if Z_index >= z_dim_FieldMatrix:
                Z_index = z_dim_FieldMatrix-1
            # Check if the indices are within the matrix bounds
            if 0 <= X_index < x_dim_FieldMatrix and 0 <= Y_index < y_dim_FieldMatrix and 0 <= Z_index < z_dim_FieldMatrix:
                # Get the electric field vector at the current carrier position
                electric_field_value = self.electric_field_matrix[X_index, Y_index, Z_index]
            else:
                electric_field_value = self.electric_field_matrix[-1, -1, -1]
            # Update the position based on the electric field
            carrier[0] += electric_field_value[0] * TimeStep_ms / 1000  # Increment in the x direction
            carrier[1] += electric_field_value[1] * TimeStep_ms / 1000  # Increment in the y direction
            carrier[2] += electric_field_value[2] * TimeStep_ms / 1000  # Increment in the z direction
            # Check if the carrier has reached the drain
            if carrier[0] >= Terminate_x:
                charge_carriers.remove(carrier)
                
                
        # Generate new charge carriers near the source (with a limit based on setting: MaxRenderedCarriers)
        for i in range(int(carriers_to_inject)):
            if len(charge_carriers) < MaxRenderedCarriers:
                spawn_y = random.uniform(self.dev_physical_parameters.min_y_gate_oxide, self.dev_physical_parameters.min_y_gate_oxide - VisualChannelThickness) + y_offset  # Randomize the y position near the source
                spawn_z = random.uniform(Spawn_z_min, Spawn_z_max)   # ToDo: Change this to use min and max of the source! Randomize the z position near the source
                charge_carriers.append([Spawn_Face_X, spawn_y, spawn_z])
        # Handle the fractional part of new carriers with a probability:
        fractional_part = carriers_to_inject % 1 # Get fractional part
        if random.random() < fractional_part and len(charge_carriers) < MaxRenderedCarriers:
            Spawn_y = random.uniform(self.dev_physical_parameters.min_y_gate_oxide, self.dev_physical_parameters.min_y_gate_oxide - VisualChannelThickness) + y_offset
            Spawn_z = random.uniform(Spawn_z_min, Spawn_z_max)  # Randomize the z position near the source
            charge_carriers.append([Spawn_Face_X, Spawn_y, Spawn_z])
        # Force a repaint after updating charge carriers
        self.update()

        
    def toggle_axes(self):
        self.show_axes = not self.show_axes
        self.update()
        # Update the parameter label with the current axis colors
        if self.show_axes:
            # ToDo: get this from settings
            self.parent().paramLabel.setText("Axis Colors:\nRed=X, Green=Y, Blue=Z")
        else:
            self.parent().paramLabel.clear()
    
    def toggle_electric_field(self):
        # Method to toggle the electric field attribute and trigger widget update
        self.show_electric_field = not self.show_electric_field
        self.update()
        # Update the field label with the current axis colors
        if self.show_electric_field:
            #ToDo: get this value from settings
            self.parent().electricFieldLabel.setText("Field Color: Orange")
        else:
            self.parent().electricFieldLabel.clear()

    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)  # Enable blending for transparency
        glDisable(GL_CULL_FACE)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)  # Set the blending function
        glClearColor(0.0, 0.0, 0.0, 1.0)

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        # Ensure that one of the dimensions is cast to a float for floating-point division
        aspect_ratio = float(w) / h
        gluPerspective(60, aspect_ratio, 0.01, 100.0)
        glMatrixMode(GL_MODELVIEW)

    def paintGL(self):
        # This big function paints the device, carriers, etc.
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        glTranslatef(0.0, 0.0, -5.0 * self.zoom_factor)  # Adjust zoom
        glTranslatef(self.pan_x, self.pan_y, 0.0)  # Apply pan
        glRotatef(self.rotationX, 1, 0, 0)
        glRotatef(self.rotationY, 0, 1, 0)
        # Define faces for blocks of the device
        faces = [
            (0, 1, 2, 3),
            (3, 2, 6, 7),
            (7, 6, 5, 4),
            (4, 5, 1, 0),
            (1, 5, 6, 2),
            (4, 0, 3, 7)
        ]
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]        
        # Draw charge carriers as spheres (draw before rest of device - otherwise GL has issues with transparancy)
        glColor4f(0.4, 0.9, 1, 1)  # light blue color with full opacity
        glPointSize(5.0)  # point size of carriers
        glBegin(GL_POINTS)
        for carrier in charge_carriers:
            glVertex3fv(carrier)
        glEnd()
        
        #############################################################################################
        #Start of drawing electric field:
        if self.show_electric_field:
            #ToDo: calculate electric field, and then exaggerate by a factor of ExaggerateChannel
            # Set the color for the electric field vectors
            glColor4f(1.0, 0.7, 0.0, 1.0)  # Yellow color with full opacity
            # Get the dimensions of the electric field matrix
            x_dim, y_dim, z_dim, _ = self.electric_field_matrix.shape
            #Calculate start of field:
            ChannelThicknessLocal = self.dev_physical_parameters.max_depletion_region_width_microns*ExaggerateChannel
            x_start = x_offset + self.dev_physical_parameters.source_width + self.dev_physical_parameters.min_x_source
            y_start = y_offset + self.dev_physical_parameters.min_y_gate_oxide - ChannelThicknessLocal
            z_start = z_offset + self.dev_physical_parameters.min_z_source
            # Loop through the electric field matrix and draw vectors at each point
            for x in range(x_dim):
                for y in range(y_dim):
                    if DrawFieldOnlyOnce == 1:
                        z = 0 # lock only one plane
                        # Get the electric field vector at this point
                        electric_field_value = self.electric_field_matrix[x, y, z]
                        # Translate and scale as needed:
                        x_with_offset = x_start + (x * FieldResolutionX) + 0.5*FieldResolutionX  # Apply x offset and scaling
                        y_with_offset = y_start + (y * FieldResolutionY) + 0.5*FieldResolutionY  # Apply y offset and scaling
                        z_with_offset = z_start + (z * FieldResolutionZ) + ((z_dim+1)/2)*FieldResolutionZ  # Apply z offset and scaling
                        # Create a list with the three values
                        resolutions = [FieldResolutionX, FieldResolutionY, FieldResolutionZ]
                        # Find the second smallest value
                        FieldResolutionDraw = sorted(resolutions)[1]
                        # Calculate the endpoint of the vector
                        endpoint = (np.array([x_with_offset, y_with_offset, z_with_offset]) + field_Vector_Draw_scale * electric_field_value*FieldResolutionDraw)
                        # Draw the line from the current point to the endpoint
                        glBegin(GL_LINES)
                        glVertex3f(x_with_offset, y_with_offset, z_with_offset)
                        glVertex3fv(endpoint)
                        glEnd()
                        # Draw an arrowhead at the endpoint
                        arrow_length = 1.2*FieldResolutionDraw  # Arrow length
                        arrowhead_size = 0.2*FieldResolutionDraw  # Size of the arrowhead
                        arrowhead_dir = (endpoint - np.array([x_with_offset, y_with_offset, z_with_offset])) * arrowhead_size / arrow_length
                        glBegin(GL_TRIANGLES)
                        glVertex3fv(endpoint)
                        glVertex3fv(endpoint - arrowhead_dir + np.array([arrowhead_dir[1], -arrowhead_dir[0], 0]))
                        glVertex3fv(endpoint - arrowhead_dir - np.array([arrowhead_dir[1], -arrowhead_dir[0], 0]))
                        glEnd()
                    else:
                        for z in range(z_dim):
                            # Get the electric field vector at this point
                            electric_field_value = self.electric_field_matrix[x, y, z]
                            # Translate and scale as needed:
                            x_with_offset = x_start + (x * FieldResolutionX) + 0.5*FieldResolutionX  # Apply x offset and scaling
                            y_with_offset = y_start + (y * FieldResolutionY) + 0.5*FieldResolutionY  # Apply y offset and scaling
                            z_with_offset = z_start + (z * FieldResolutionZ) + 1*FieldResolutionZ  # Apply z offset and scaling
                            # Create a list with the three values
                            resolutions = [FieldResolutionX, FieldResolutionY, FieldResolutionZ]
                            # Find the second smallest value
                            FieldResolutionDraw = sorted(resolutions)[1]
                            # Calculate the endpoint of the vector
                            endpoint = (np.array([x_with_offset, y_with_offset, z_with_offset]) + field_Vector_Draw_scale * electric_field_value*FieldResolutionDraw)
                            # Draw the line from the current point to the endpoint
                            glBegin(GL_LINES)
                            glVertex3f(x_with_offset, y_with_offset, z_with_offset)
                            glVertex3fv(endpoint)
                            glEnd()
                            # Draw an arrowhead at the endpoint
                            arrow_length = 1.2*FieldResolutionDraw  # Arrow length
                            arrowhead_size = 0.2*FieldResolutionDraw  # Size of the arrowhead
                            arrowhead_dir = (endpoint - np.array([x_with_offset, y_with_offset, z_with_offset])) * arrowhead_size / arrow_length
                            glBegin(GL_TRIANGLES)
                            glVertex3fv(endpoint)
                            glVertex3fv(endpoint - arrowhead_dir + np.array([arrowhead_dir[1], -arrowhead_dir[0], 0]))
                            glVertex3fv(endpoint - arrowhead_dir - np.array([arrowhead_dir[1], -arrowhead_dir[0], 0]))
                            glEnd()
                        
        #End of draw field
        #############################################################################################
        #Render the device itself:
        for part_data in self.device_parts:
            part_vertices = part_data["vertices"]
            part_color = part_data["color"]
            # Draw the edges of the extruded rectangle (white, semi-opaque)
            glColor4f(1.0, 1.0, 1.0, 0.6)  # Set alpha component to 0.7 for semi-opacity
            glBegin(GL_LINES)
            for edge in edges:
                for vertex_index in edge:
                    vertex = part_vertices[vertex_index]
                    # Add the fixed offsets to the vertex
                    adjusted_vertex = [
                        vertex[0] + x_offset,
                        vertex[1] + y_offset,
                        vertex[2] + z_offset
                    ]
                    glVertex3fv(adjusted_vertex)
            glEnd()
            # Draw the faces of the extruded rectangle (use the part's color and make it semi-opaque)
            glColor4f(*part_color, 0.5)  # Use the part's color with alpha component set to 0.5 for semi-opacity
            glBegin(GL_QUADS)
            for face in faces:
                for vertex_index in face:
                    vertex = part_vertices[vertex_index]
                    # Add the fixed offsets to the vertex
                    adjusted_vertex = [
                        vertex[0] + x_offset,
                        vertex[1] + y_offset,
                        vertex[2] + z_offset
                    ]
                    glVertex3fv(adjusted_vertex)
            glEnd()
         
        
        ##################################################################################################
        #Draw the axes until next section break:
        if self.show_axes:
            # Draw the axes (x, y, z) with arrows
            axis_length = 2.0  # Adjust the length of the axes as needed
            # X-axis (red)
            glColor4f(1.0, 0.0, 0.0, 1.0)  # Red color with full opacity
            glBegin(GL_LINES)
            glVertex3f(0.0, 0.0, 0.0)
            glVertex3f(axis_length, 0.0, 0.0)
            glEnd()
            # Y-axis (green)
            glColor4f(0.0, 1.0, 0.0, 1.0)  # Green color with full opacity
            glBegin(GL_LINES)
            glVertex3f(0.0, 0.0, 0.0)
            glVertex3f(0.0, axis_length, 0.0)
            glEnd()
            # Z-axis (blue)
            glColor4f(0.0, 0.0, 1.0, 1.0)  # Blue color with full opacity
            glBegin(GL_LINES)
            glVertex3f(0.0, 0.0, 0.0)
            glVertex3f(0.0, 0.0, axis_length)
            glEnd()
            # Add arrows at the ends of the axes
            arrow_length = 0.1  # Length of the arrowheads
            arrow_width = 0.05  # Width of the arrowheads
            # X-axis arrow
            glColor4f(1.0, 0.0, 0.0, 1.0)  # Red color with full opacity
            glBegin(GL_TRIANGLES)
            glVertex3f(axis_length, 0.0, 0.0)
            glVertex3f(axis_length - arrow_length, arrow_width, 0.0)
            glVertex3f(axis_length - arrow_length, -arrow_width, 0.0)
            glEnd()
            # Y-axis arrow
            glColor4f(0.0, 1.0, 0.0, 1.0)
            glBegin(GL_TRIANGLES)
            glVertex3f(0.0, axis_length, 0.0)
            glVertex3f(arrow_width, axis_length - arrow_length, 0.0)
            glVertex3f(-arrow_width, axis_length - arrow_length, 0.0)
            glEnd()
            # Z-axis arrow
            glColor4f(0.0, 0.0, 1.0, 1.0)
            glBegin(GL_TRIANGLES)
            glVertex3f(0.0, 0.0, axis_length)
            glVertex3f(0.0, arrow_width, axis_length - arrow_length)
            glVertex3f(0.0, -arrow_width, axis_length - arrow_length)
            glEnd()
        #End of draw axes
           



    def mousePressEvent(self, event):
        if event.buttons() == Qt.RightButton:
            self.right_mouse_button_pressed = True
            self.last_right_mouse_pos = event.pos()
        elif event.buttons() == Qt.LeftButton:
            self.lastPos = event.pos()

    def mouseMoveEvent(self, event):
        if self.right_mouse_button_pressed:
            if self.last_right_mouse_pos:
                dx = event.x() - self.last_right_mouse_pos.x()
                dy = event.y() - self.last_right_mouse_pos.y()
                self.last_right_mouse_pos = event.pos()
                self.pan(dx, dy)
        elif self.lastPos:
            dx = event.x() - self.lastPos.x()
            dy = event.y() - self.lastPos.y()
            self.rotationX += dy
            self.rotationY += dx
            self.update()
            self.lastPos = event.pos()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.RightButton:
            self.right_mouse_button_pressed = False
        elif event.button() == Qt.LeftButton:
            self.lastPos = None

    def wheelEvent(self, event):
        # Zoom in/out based on the mouse scroll direction
        delta = event.angleDelta().y()
        zoom_speed = 0.1  # Adjust the zoom speed as needed
        self.zoom_factor += zoom_speed if delta > 0 else -zoom_speed
        self.zoom_factor = max(0.1, self.zoom_factor)  # Ensure a minimum zoom level
        self.update()

    def reset_view(self):
        # Reset the view to its default starting position
        self.rotationX = 0
        self.rotationY = 0
        self.zoom_factor = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.update()

    def pan(self, dx, dy):
        # Pan the scene based on right-click drag
        pan_speed = 0.01  # Adjust the pan speed as needed
        self.pan_x += dx * pan_speed
        self.pan_y -= dy * pan_speed
        self.update()

class DeviceLoader:
    def __init__(self):
        # Initialize device information
        self.device_parts = {}  # Dictionary to store device part information
        self.source_face = None  # Face where the source is located
        self.drain_face = None   # Face where the drain is located
        self.gate_dielectric = None  # Area where the gate dielectric (oxide) is located
        self.body_region = None  # The region between source and drain
        self.gate = None #The gate region
        # Create an instance to hold physical parameters
        self.physical_parameters = DevicePhysicalParameters()

    def remove_comments_from_json(self, json_string):
        # This function removes comments from the JSON file
        lines = json_string.split('\n')
        filtered_lines = [line for line in lines if not line.strip().startswith('#')]
        return '\n'.join(filtered_lines)

    def calculate_device_parameters(self):
        # This function is meant to calculate the required device parameters based on the device physics. 
        #ToDo (see next comment): Can we split this function into two, one for startup (parameters that are fixed)
        #Actually, about the above ToDo, we could simply update values in the function: calculate_DC_operating_point()
        #And another for dynamic parameters such as Vth which is modified as you go?
        # Calculate the width of the source based on vertices
        source_vertices = self.source_face
        self.physical_parameters.min_x_source = min(vertex[0] for vertex in source_vertices)
        self.physical_parameters.max_x_source = max(vertex[0] for vertex in source_vertices)
        self.physical_parameters.source_width = self.physical_parameters.max_x_source - self.physical_parameters.min_x_source
        # Calculate the minimum X value of the drain vertices
        drain_vertices = self.drain_face
        self.physical_parameters.min_x_drain = min(vertex[0] for vertex in drain_vertices)
        # Calculate the distance between source and drain
        self.physical_parameters.L_SourcetoDrain_Microns = self.physical_parameters.min_x_drain - self.physical_parameters.max_x_source
        # Calculate the minimum and maximum Y dimensions of the gate oxide
        gate_oxide_vertices = self.gate_dielectric
        self.physical_parameters.min_y_gate_oxide = min(vertex[1] for vertex in gate_oxide_vertices)
        self.physical_parameters.max_y_gate_oxide = max(vertex[1] for vertex in gate_oxide_vertices)
        self.physical_parameters.min_y_SourceDrain = min(vertex[1] for vertex in source_vertices)
        self.physical_parameters.min_z_source = min(vertex[2] for vertex in source_vertices)
        self.physical_parameters.max_z_source = max(vertex[2] for vertex in source_vertices)
        self.physical_parameters.W_DeviceWidth_Microns = self.physical_parameters.max_z_source - self.physical_parameters.min_z_source
        #Now let's calculate some device physics parameters, first Fermi potential:
        body_doping_type = self.physical_parameters.body_doping_type
        if body_doping_type == "p-type":
            p = self.physical_parameters.body_doping_concentration*(1e6) #In Si units
            self.physical_parameters.fermi_potential = (uc_k * DeviceTemperature / uc_q) * math.log10(p / (Silicon_ni*(1e6)))
        elif body_doping_type == "n-type":
            n = self.physical_parameters.body_doping_concentration*(1e6) #In Si units
            self.physical_parameters.fermi_potential = (uc_k * DeviceTemperature / uc_q) * math.log10((Silicon_ni*(1e6)) / n)
        else:
            print("Could not compute fermi level, check device file for errors")
        #Let's also calculate the depletion region maximum width:
        EpsSi_local = uc_EpsFS*EpsSi #Permitivity of silicon F/m
        FermiPot = self.physical_parameters.fermi_potential #get fermi potential
        DopantLevel = self.physical_parameters.body_doping_concentration*(1e6) #doping N in Si units
        Wmax = math.sqrt((4*EpsSi_local*abs(FermiPot)) / (uc_q * DopantLevel)) #Final units Si
        self.physical_parameters.max_depletion_region_width_microns = Wmax*(1e6) #Units in um
        #Now for the oxide capacitance: (This is often called C0 in text - a fixed oxide capacitance, ignoring inversion)
        gateDielectricThickness = (self.physical_parameters.max_y_gate_oxide - self.physical_parameters.min_y_gate_oxide)/(1e6) #In Si units
        oxideCap_local = (EpsSiOx*uc_EpsFS)/(gateDielectricThickness) #Oxide capacitance in Si units
        self.physical_parameters.oxide_capacitance_per_unit_area_nFperCMsquare = oxideCap_local*(1e5) #units in nF/cm^2
        #ToDo: Calculate gate capacitance based on Vgs, see Sze page 192
        #Calculating the threshold voltage here:
        if body_doping_type == "p-type":
            _p = self.physical_parameters.body_doping_concentration*(1e6) #In Si units
            _devGamma = (math.sqrt(2*EpsSi_local*uc_q*_p)/oxideCap_local)
            self.physical_parameters.devGamma_SQRT_Volts = _devGamma
            threshold_local = ((_devGamma*(math.sqrt(abs(2*FermiPot)))) + abs(2*FermiPot))
            self.physical_parameters.threshold_voltage_V = threshold_local
        elif body_doping_type == "n-type":
            _n = self.physical_parameters.body_doping_concentration*(1e6) #In Si units
            _devGamma = (math.sqrt(2*EpsSi_local*uc_q*_n)/oxideCap_local)
            self.physical_parameters.devGamma_SQRT_Volts = _devGamma
            threshold_local = -((_devGamma*(math.sqrt(abs(2*FermiPot)))) + abs(2*FermiPot))
            self.physical_parameters.threshold_voltage_V = threshold_local
        else:
            print("Could not compute threshold voltage, check device file for errors")
        #We can also use an approximate carrier mobility for now
        #ToDo: change this to a proper calculation based on temperature, doping, etc (skeleton already setup):
        if body_doping_type == "p-type":
            self.physical_parameters.channel_carrier_mobility_cmSquarePerVoltSec = 400
        elif body_doping_type == "n-type":
            self.physical_parameters.channel_carrier_mobility_cmSquarePerVoltSec = 1200
        else:
            print("Could not compute mobility, check device file for errors in body doping type")
        
        
    def load_device_info(self, device_info_file_path):
        try:
            with open(device_info_file_path, "r") as file:
                json_content = file.read()
                json_content = self.remove_comments_from_json(json_content)
                device_info_data = json.loads(json_content)
            # Load device information from the JSON data
            self.device_parts = device_info_data["device_parts"]
            # Set the source and drain faces based on the device_parts data
            # For example, find the face named "Source" and assign it to self.source_face
            source_face_name = "Source"
            drain_face_name = "Drain"
            gate_dielectric_name = "GateOxide"
            gate_face_name = "Gate"
            body_region_name = "Body"
            for part_data in self.device_parts:
                part_name = part_data["name"]
                part_vertices = part_data["vertices"]
                if part_name == source_face_name:
                    self.source_face = part_vertices
                elif part_name == drain_face_name:
                    self.drain_face = part_vertices
                elif part_name == gate_dielectric_name:
                    self.gate_dielectric = part_vertices
                elif part_name == body_region_name:  # Load the body region
                    self.body_region = part_vertices
                elif part_name == gate_face_name:  # Load the gate region
                    self.gate = part_vertices
                if "doping" in part_data:
                    doping_data = part_data["doping"]
                    try:
                        if part_name == "Source":
                            self.physical_parameters.source_doping_type = doping_data["type"]
                            self.physical_parameters.source_doping_concentration = doping_data["concentration"]
                        elif part_name == "Drain":
                            self.physical_parameters.drain_doping_type = doping_data["type"]
                            self.physical_parameters.drain_doping_concentration = doping_data["concentration"]
                        elif part_name == "Gate":
                            self.physical_parameters.channel_doping_type = doping_data["type"]
                            self.physical_parameters.channel_doping_concentration = doping_data["concentration"]
                        elif part_name == "Body":  # Load doping information for the body region
                            self.physical_parameters.body_doping_type = doping_data["type"]
                            self.physical_parameters.body_doping_concentration = doping_data["concentration"]
                    except KeyError as e:
                        print(f"Error loading doping data for {part_name}: {e}")
            # Calculate the device parameters
            self.calculate_device_parameters()
        except FileNotFoundError:
            print("Device information file not found.")
        except Exception as e:
            print(f"Error loading device information: {e}")


class DevicePhysicalParameters:
    #This class defines the device physical parameters such as distance from source to drain,
    #min and max x positions for charge carriers, etc. To add a value here, simply add a line
    #for example: self.mobility_channel = 0.1
    #Then go to the function calculate_device_parameters and calculate a starting value, or
    #leave it as default.
    def __init__(self):
        self.source_width = 0.0
        self.min_y_gate_oxide = 0.0
        self.max_y_gate_oxide = 0.0
        self.min_x_source = 0.0
        self.min_x_drain = 0.0
        self.min_y_SourceDrain = 0.0
        self.W_DeviceWidth_Microns = 0.0 #Device parameter W
        self.L_SourcetoDrain_Microns = 0.0 #Device parameter L
        self.max_z_source = 0.0
        self.min_z_source = 0.0
        # Doping information
        self.source_doping_type = ""  # Type of source doping (e.g., "n-type" or "p-type")
        self.source_doping_concentration = 0.0  # Doping concentration in cm^-3
        self.drain_doping_type = ""  # Type of drain doping (e.g., "n-type" or "p-type")
        self.drain_doping_concentration = 0.0  # Doping concentration in cm^-3
        self.body_doping_type = ""  # Type of channel doping (e.g., "n-type" or "p-type")
        self.body_doping_concentration = 0.0  # Doping concentration in cm^-3
        # Device Physics information
        self.fermi_potential = 0.3 #The fermi level of the silicon interface
        self.max_depletion_region_width_microns = 0.01 #Depletion region width in microns
        self.oxide_capacitance_per_unit_area_nFperCMsquare = 430e-9 #Capacitance in Farad per square centimeter
        self.threshold_voltage_V = 0.4 #The threshold voltage in Volts
        self.channel_carrier_mobility_cmSquarePerVoltSec = 800 #Channel mobility in units of (cm^2)/(V*s)
        self.devGamma_SQRT_Volts = 0 #Sqrt(2Es*q*Na)/Cox #ToDo change with typical value
        #ToDo: add , Vt, etc into here, as well as in calculate_device_parameters()
        # Operating point and DC parameters:
        self.Vgs_V = 0 #Gate to source voltage
        self.Vds_V = 0 #Drain to source voltage
        self.Vdsat_V = 0 #Vdsat in Volts
        self.Idrain_uA = 10 #Charge carrier current in microAmps
        self.lambda_PerV = 0.0 # Output impedance constant in V^-1
        self.OperatingRegion = 0 # User info labels/text for operating region
        # Small Signal Parameters:
        self.gm_uS = 0 # Small signal transconductance of Vgs
        self.gds_uS = 0 # Small signal transconductance of Vds
        

class ParameterDisplayDialog(QDialog):
    #This dialog box that pops up is meant to show the user details about the device such as
    #fermi levels and threshold voltages - anything we read from the file or calculate
    def __init__(self, device_parameters, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Device Parameters")
        self.setGeometry(200, 200, 400, 500)
        self.device_parameters = device_parameters
        # Create a scroll area to contain your parameter display
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        # Create a widget to hold the parameter information
        parameters_widget = QWidget()
        parameters_layout = QVBoxLayout(parameters_widget)
        # Create a label to display the parameters
        self.paramLabel = QLabel(self)
        self.paramLabel.setWordWrap(True)
        parameters_layout.addWidget(self.paramLabel)
        self.scroll_area.setWidget(parameters_widget)
        self.update_parameters_text()
        #self.setLayout(parameters_layout)
        
    def resizeEvent(self, event):
        # When the dialog is resized, adjust the scroll area's size to match the dialog
        self.scroll_area.setGeometry(0, 0, self.width(), self.height())
    
    def update_parameters_text(self):
        # Generate a string with the device parameters
        # Here we can add more stuff to display
        parameters_text = "Device Parameters:\n"
        parameters_text += f"Source Doping Type: {self.device_parameters.source_doping_type}\n"
        parameters_text += f"Source Doping Concentration: {self.device_parameters.source_doping_concentration}\n"
        parameters_text += f"Drain Doping Type: {self.device_parameters.drain_doping_type}\n"
        parameters_text += f"Drain Doping Concentration: {self.device_parameters.drain_doping_concentration}\n"
        parameters_text += f"Channel Doping Type: {self.device_parameters.body_doping_type}\n"
        parameters_text += f"Channel Doping Concentration: {self.device_parameters.body_doping_concentration}\n"
        parameters_text += f"Fermi Level: {self.device_parameters.fermi_potential:.3f} V\n"
        parameters_text += f"Depletion Region W: {self.device_parameters.max_depletion_region_width_microns: .3f} um\n"
        parameters_text += f"Gate Capacitance C0: {self.device_parameters.oxide_capacitance_per_unit_area_nFperCMsquare: .3e} nF/cm^2\n"
        parameters_text += f"Threshold Voltage: {self.device_parameters.threshold_voltage_V: .3f} V\n"
        parameters_text += f"Channel Mobility: {self.device_parameters.channel_carrier_mobility_cmSquarePerVoltSec: .3f} cm^2/V.s\n"
        parameters_text += f"\nDC Operating Point:\n"
        parameters_text += f"Gate to Source Voltage (Vgs): {self.device_parameters.Vgs_V: .3f} V\n"
        parameters_text += f"Drain to Source Voltage (Vds): {self.device_parameters.Vds_V: .3f} V\n"
        parameters_text += f"Drain Current (Ids): {self.device_parameters.Idrain_uA: .3f} uA\n"
        # List of region names corresponding to numeric codes
        region_names = ["Cut-off", "Triode", "Saturation", "Subthreshold", "Breakdown"]
        # Get the region code from self.device_parameters.OperatingRegion
        region_code = int(self.device_parameters.OperatingRegion)
        # Convert the region code to the corresponding region name
        region_name = region_names[region_code] if 0 <= region_code < len(region_names) else "Unknown"
        # Append the region information to the parameters_text
        parameters_text += f"Device Region: {region_code} - {region_name}\n"
        parameters_text += f"Impedance Constant (lambda): {self.device_parameters.lambda_PerV: .3f} /V\n"
        parameters_text += f"Transconductance (gm): {self.device_parameters.gm_uS: .3f} uS\n"
        parameters_text += f"DS Transconductance (gds): {self.device_parameters.gds_uS: .3f} uS\n"
        self.paramLabel.setText(parameters_text)        
        
        
class IV_Curve_MatplotlibWidget(QWidget):
    def __init__(self, device, parent=None):
        super().__init__(parent)
        self.device = device
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.canvas = FigureCanvas(self.fig)        
        self.fig.patch.set_facecolor((0.5, 0.5, 0.5))
        self.fig.patch.set_alpha(0.0)
        self.ax.set_facecolor((0.5, 0.5, 0.5, 0.0))  # Set background color to black
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.toolbar.setStyleSheet("background-color: rgba(255, 255, 255, 0.5); border: 0px;")
        # Set the size of the FigureCanvas
        self.canvas.setMinimumSize(200, 250)  # Set minimum size
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.addWidget(self.toolbar)
        self.setLayout(layout)
        self.plot_surface()
        # Setup plot with required labels and visuals:
        label_color = 'yellow'
        self.ax.set_xlabel('Vds (V)', color=label_color)
        self.ax.set_zlabel('Id (uA)', color=label_color)
        self.ax.set_ylabel('Vgs (V)', color=label_color)
        self.ax.tick_params(axis='x', colors=label_color)
        self.ax.tick_params(axis='y', colors=label_color)
        self.ax.tick_params(axis='z', colors=label_color)
        # lock axes to make it easier to understand:
        self.ax.set_xlim([min_vds, max_vds])
        self.ax.set_zlim([min_current_uA, max_current_uA])
        self.ax.set_ylim([min_vgs, max_vgs])
        # Set up a timer to update the plot every second

    def plot_surface(self):
        vgs_range = np.linspace(min_vgs, max_vgs, 25)
        vds_range = np.linspace(min_vds, max_vds, 25)
        Vgs, Vds = np.meshgrid(vgs_range, vds_range)
        # Ensure that Vds, Vgs, and Idrain are 2D arrays
        Vds = np.atleast_2d(Vds)
        Vgs = np.atleast_2d(Vgs)
        # Initialize Idrain array
        Idrain = np.zeros_like(Vgs)
        # Initialize region array
        region = np.zeros_like(Vgs)
        # Iterate over each element and compute Idrain and region
        for i in range(Vgs.shape[0]):
            for j in range(Vgs.shape[1]):
                Vgs_ij = Vgs[i, j]
                Vds_ij = Vds[i, j]
                Idrain[i, j], _, region[i, j], _, _, _ = Device_Mathematical_Model(Vgs_ij, Vds_ij, self.device)
        # Convert region to integers
        region = region.astype(int)
        colors = custom_cmap_DC_Plot(region)
        # Plot the surface with different colors for different regions
        self.surface_plot = self.ax.plot_surface(
            Vds, Vgs, Idrain, facecolors=colors,
            edgecolor='0.3', linewidth=0.15, alpha=0.4
        )
        # Get the colors from the colormap for the legend
        legend_colors = custom_cmap_DC_Plot(np.arange(custom_cmap_DC_Plot.N))
        # Create legend patches using the colormap colors
        legend_patches = [Patch(color=legend_colors[i], label=label)
                          for i, label in enumerate(['Cutoff', 'Linear', 'Saturation'])]
        # Add legend
        self.ax.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(1.35, 1.15))
        self.ax.autoscale()
        # Redraw the canvas
        self.canvas.draw()
        

    def Refresh_DC_OP(self):
        # Update the position of the DC operating point
        operating_vds = self.device.Vds_V
        operating_drain_Current = self.device.Idrain_uA
        operating_vgs = self.device.Vgs_V
        # Remove the previous scatter plot if it exists
        if hasattr(self, 'dc_point'):
            self.dc_point.remove()
        # Get the colors from the colormap for the legend
        legend_colors = custom_cmap_DC_Plot(np.arange(custom_cmap_DC_Plot.N))
        # Create legend patches using the colormap colors
        legend_patches = [Patch(color=legend_colors[i], label=label)
                          for i, label in enumerate(['Cutoff', 'Linear', 'Saturation'])]
        # Plot the updated DC operating point
        self.dc_point = self.ax.scatter(
            operating_vds,
            operating_vgs,
            operating_drain_Current,
            color='cyan', label='DC OP',
            s=30
        )
        # Add the DC operating point to the existing legend_patches
        legend_patches.append(Patch(color='cyan', label='DC OP'))
        # Update the legend with the modified labels
        self.ax.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(1.35, 1.15))
        self.canvas.draw()



# This is the heart of the program - the mathematical model which we use to represent
# the mosfet. Feel free to change out as you'd like, but I have been using a
# model from Sze up to now.
def Device_Mathematical_Model(Vgs, Vds, device):
    # Pass the vgs and vds values explicitly, so that other operating points can be computed,
    # without needing to discturb the current operating conditions.
    # This model is presently a sort of 'fit' to make it visually appealing. Better model needed...
    #ToDo, also calculate using square law and compare... Also operating region (linear, sat, etc)
    #ToDo: built in hard limits to the functions, so it sticks to the areas where it is valid
    _w = device.W_DeviceWidth_Microns / (1e6)
    _l = device.L_SourcetoDrain_Microns / (1e6)
    _mu = device.channel_carrier_mobility_cmSquarePerVoltSec / (1e4)
    _cox = device.oxide_capacitance_per_unit_area_nFperCMsquare * (1e-5)
    _fermipot = device.fermi_potential
    _devGamma = device.devGamma_SQRT_Volts
    _Beta = (_w / _l) * _mu * _cox
    _lambda = 0 # Set to zero for if we are not in saturation
    _gds_uS = 0 # Set to zero for not sat
    Vth = device.threshold_voltage_V
    Vdsat = abs(Vgs-Vth)/1.85 # value seems best for a good visual fit... This defines switchover from linear to sat
    Delta_Vgs = Vgs*(1/numerical_Differentiation_Resolution)
    Vgs2_Deriv = Vgs + Delta_Vgs
    Delta_Vds = Vds*(1/numerical_Differentiation_Resolution)
    Vds2_Deriv = Vds + Delta_Vds
    OperatingRegion = 5 #Set to unknown until better defined
    _N_Body = device.body_doping_concentration*(1e6)
    _kds = math.sqrt(abs((2*EpsSi*uc_EpsFS)/(uc_q*_N_Body))) # Absolute value just for safety, but it should anyways always be positive
    # Calculate device operating region:
    # 0 cut-off, 1 triode, 2 sat, 3 subth, 4 breakdown, 5 unkown/error
    try:
        if (Vgs < Vth ) or (Vgs == 0) or (Vds == 0) :
            OperatingRegion = 0  # Cut-off
            Idrain = 0
            gm = 0
        elif Vgs >= Vth and 0 < Vds < (Vdsat):
            OperatingRegion = 1  # Triode/Linear
            # The below equation is from Sze equation (86) page 205:
            Idrain = _Beta * (((Vgs - (2 * _fermipot) - (Vds / 2)) * Vds) - ((2 / 3) * _devGamma * (
                    ((Vds + (2 * _fermipot)) ** (3 / 2)) - ((2 * _fermipot) ** (3 / 2)))))
            Idrain2 = _Beta * (((Vgs2_Deriv - (2 * _fermipot) - (Vds / 2)) * Vds) - ((2 / 3) * _devGamma * (
                    ((Vds + (2 * _fermipot)) ** (3 / 2)) - ((2 * _fermipot) ** (3 / 2)))))
            Idrain3 = _Beta * (((Vgs - (2 * _fermipot) - (Vds2_Deriv / 2)) * Vds2_Deriv) - ((2 / 3) * _devGamma * (
                    ((Vds2_Deriv + (2 * _fermipot)) ** (3 / 2)) - ((2 * _fermipot) ** (3 / 2)))))
            # Calculate gm as slope of Id/Vgs
            gm = (Idrain2-Idrain)/Delta_Vgs
            # Now to compute gds as slope of Id/Vds
            _gds_uS = ((Idrain3-Idrain)/Delta_Vds)*(1e6)
        elif (Vgs >= Vth) and Vds > (Vdsat):
            OperatingRegion = 2  # Saturation
            #ToDo: this whole model is kind of a fit to make it visually correct, but it is terribly wrong
            #fundamentally. We should maybe go for a 3rd order function at some point
            num_points = 100  # Number of points to evaluate between min/max vds
            # The next couple of lines are for computing lambda - the output impedance constant :
            # Computing the output impedance (Johns & Martin Pg 27)
            # ToDo: Get below working like it should
            _lambda = _kds/(2*_l*math.sqrt(abs(Vds-Vgs+Vth+_fermipot)))
            _lambda = 0.08 # override for now
            _gds_uS = (_lambda*(_Beta/2)*(Vgs-Vth)**2)*(1e6)
            #Below is sort of a fit, where we find the highest Id for the given vgs, by sweeping vds.
            #this is kind of how saturation works, but is a bit simplified
            current_values_1 = [
                _Beta * (((Vgs - (2 * _fermipot) - (vds_ / 2)) * vds_) - ((2 / 3) * _devGamma * (
                    ((vds_ + (2 * _fermipot)) ** (3 / 2)) - ((2 * _fermipot) ** (3 / 2)))))
                for vds_ in (min_vds + i * (max_vds - min_vds) / (num_points - 1) for i in range(num_points))
            ]
            current_values_2 = [
                _Beta * (((Vgs2_Deriv - (2 * _fermipot) - (vds_ / 2)) * vds_) - ((2 / 3) * _devGamma * (
                    ((vds_ + (2 * _fermipot)) ** (3 / 2)) - ((2 * _fermipot) ** (3 / 2)))))
                for vds_ in (min_vds + i * (max_vds - min_vds) / (num_points - 1) for i in range(num_points))
            ]
            # Multiply the current values by the additional factor
            current_values_1 = [current * (1 + (_lambda * (Vds - (Vdsat)))) for current in current_values_1]
            current_values_2 = [current * (1 + (_lambda * (Vds - (Vdsat)))) for current in current_values_2]
            # Find the maximum current value
            Idrain = max(current_values_1)
            Idrain2 = max(current_values_2)
            # Calculate gm as slope of Id/Vgs
            gm = (Idrain2-Idrain)/Delta_Vgs
        elif (0 < Vgs < Vth) and (Vds < Vdsat):
            OperatingRegion = 3  # Subthreshold
            # The below equation is from Sze equation (86) page 205:
            Idrain = _Beta * (((Vgs - (2 * _fermipot) - (Vds / 2)) * Vds) - ((2 / 3) * _devGamma * (
                    ((Vds + (2 * _fermipot)) ** (3 / 2)) - ((2 * _fermipot) ** (3 / 2)))))
            Idrain2 = _Beta * (((Vgs2_Deriv - (2 * _fermipot) - (Vds / 2)) * Vds) - ((2 / 3) * _devGamma * (
                    ((Vds + (2 * _fermipot)) ** (3 / 2)) - ((2 * _fermipot) ** (3 / 2)))))
            # Calculate gm as slope of Id/Vgs
            gm = (Idrain2-Idrain)/Delta_Vgs
        else:
            OperatingRegion = 4  # Breakdown
            Idrain = max_current_uA/(1e6)
            gm = 0
    except Exception as e:
        print(f"An error occurred: {e}")
        OperatingRegion = 5  # Unknown
        Idrain = 0
        gm = 0
    # Check for NaN values and replace with 0
    Idrain = np.where(np.isnan(Idrain), 0, Idrain)
    gm = np.where(np.isnan(gm), 0, gm)
    # Apply hard limit to ensure Idrain is within allowed limits, and convert to uA
    Idrain_uA = max(min(Idrain * 1e6, max_current_uA), min_current_uA)
    # Convert gm to uS:
    gm_uS = gm*1e6
    # If you need, add more below. Make sure to update the main code above to
    # use or ignore these variables (ignore with a dummy variable)
    # Idrain is in uA below
    return Idrain_uA, gm_uS, OperatingRegion, _lambda, _gds_uS, Vdsat
    

# Launcher for the application
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MosfetViewer()
    window.show()
    sys.exit(app.exec_())

