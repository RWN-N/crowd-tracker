# Crowd Tracking & Recognition System

The Crowd Tracking & Recognition System is a real-time application designed to detect, track, and recognize people in crowded environments, displayed through a simple web-based interface.

---

## Prerequisites

Make sure you have the following software installed before starting:

- **Python 3.9+**  
- **Git**  
- **pip** (Python package manager)  

---

## Installation

1. **Clone the Repository**:  

   Clone the main project repository:
   ```bash
   git clone https://github.com/RWN-N/crowd-tracker.git
   cd crowd-tracker
   ```

2. **Initialize and Update Submodules**:  

   Sync the submodules and recursively update them:
   ```bash
   git submodule sync
   git submodule update --init --recursive
   ```

3. **Install Dependencies**:  

   Run the `install_requirements.py` script to install all required dependencies for the project and its submodules:  
   ```bash
   python install_requirements.py
   ```

---

## Running the Project

After installation, you can start the project by running the main script:

```bash
python main.py
```

---

## Accessing the Web Interface

Once the project is running, open your browser and access the web interface:

- **Localhost**: [http://localhost:8000](http://localhost:8000)  
- **Your IP**: Replace `localhost` with your machine's IP address.  

Example: `http://192.168.1.10:8000`

---

## Notes

- The `install_requirements.py` script ensures dependencies are installed for both the **main project** and any **submodules**.  
- Ensure the port `8000` is not blocked by firewalls or already in use.  

---
