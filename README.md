# BasicMOSFET3
Interactive Device Simulator. 

## Program Description
This simple simulator serves as an educational tool designed to aid students in understanding the workings of a MOSFET (Metal-Oxide-Semiconductor Field-Effect Transistor). Input files define the device sizing, which the simulator loads and then visualizes. The calculations of currents, voltages, etc., are based on simple second-order MOSFET model equations. All units are in microns for all axes. To use the simulator, a device file titled 'device.json' is required, defining sizing and doping parameters.

Author: Jonathan C. Rabe
LinkedIn: linkedin.com/in/jonathanrabe
Support: [Buy me a coffee](https://www.buymeacoffee.com/jonathanrabe)

## Running Steps
1. Install Python 3 (if not installed)
   - Note: Make sure to check "Add Python 3.x to PATH" during installation
   - If Python is already installed, you can skip this step
   - [Python Download](https://www.python.org/downloads/)
   - After installation, you might need to restart the command prompt or PowerShell
   - Note: Depending on how you install Python, the commands below may start with 'py' or they may start with 'python'.

2. Install required packages
   - py -3 -m pip install numpy
   - py -3 -m pip install PyQt5
   - py -3 -m pip install PyOpenGL
   - py -3 -m pip install matplotlib

3. Run BasicMOSFET3.py
   - py -3 BasicMOSFET3.py
