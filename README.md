# ThermoApps Backend Webservice
## Requirement
1. Python >= 3.8 with pip
2. SQLite3

## Preinstallation
It is highly recommend to use Python Virtual Environment!
```
cd /path/to/repo
mkdir venv

# option 1
python -m venv venv
# option 2
py -3 -m venv venv

# UNIX (Linux or macOS)
source /path/to/repo/venv/bin/activate
# Windows
/path/to/repo/venv/Scripts/activate

# to deactivate Virtual Environment
deactivate
```

## Installation
1. Install dependencies `pip install requirements.txt`
2. Create `environment.json` from`environment.example.json`
  - Defined `SECRET_KEY`, `DEBUG`, and `ALLOWED_HOST`
3. Run `python migrate.py runserver`