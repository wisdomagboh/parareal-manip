# setup.py

import subprocess 
from dm_control import suite 

#  Copy environment xml and py files into 'suite' 
_SUITE_DIR = "".join(suite.__path__)

subprocess.call(['cp', 'setup/pusher.xml', _SUITE_DIR])
subprocess.call(['cp', 'setup/pusher.py', _SUITE_DIR])
subprocess.call(['cp', 'setup/multislider.xml', _SUITE_DIR])
subprocess.call(['cp', 'setup/multislider.py', _SUITE_DIR])
subprocess.call(['cp', 'setup/multislider_render.xml', _SUITE_DIR])
subprocess.call(['cp', 'setup/multislider_render.py', _SUITE_DIR])

# Update __init_.py in 'suite' such that it contains our custom domains
subprocess.call(['cp', 'setup/__init__.py', _SUITE_DIR])