# DEPLOYMENT & INSTALLATION GUIDE
## Stock Price Forecasting System - MSc IT Project

---

## 📦 COMPLETE INSTALLATION GUIDE

### Prerequisites Check

Before starting, ensure you have:
- ✅ Python 3.10 or higher
- ✅ pip (Python package manager)
- ✅ 4GB RAM minimum (8GB recommended)
- ✅ 2GB free disk space
- ✅ Stable internet connection
- ✅ Modern web browser (Chrome, Firefox, Safari, Edge)

### Verify Python Installation

```bash
# Check Python version
python --version
# Should show: Python 3.10.x or higher

# Check pip
pip --version
# Should show pip version
```

If Python is not installed, download from: https://www.python.org/downloads/

---

## 🚀 INSTALLATION METHODS

### Method 1: Fresh Installation (Recommended)

**Step 1: Extract Project**
```bash
# Extract the zip file to your desired location
# Navigate to the project directory
cd stock_prediction_project
```

**Step 2: Create Virtual Environment**
```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate

# You should see (venv) in your terminal prompt
```

**Step 3: Install Dependencies**
```bash
# Upgrade pip first
pip install --upgrade pip

# Install all requirements
pip install -r requirements.txt

# This will take 5-10 minutes depending on your internet speed
```

**Step 4: Download NLTK Data**
```bash
# Download required NLTK packages
python -c "import nltk; nltk.download('punkt')"
python -c "import nltk; nltk.download('stopwords')"
```

**Step 5: Verify Installation**
```bash
# Run the test script
python test_system.py

# You should see all tests passing ✅
```

**Step 6: Run the Application**
```bash
# Start the Flask application
python app.py

# You should see:
# * Running on http://0.0.0.0:5000
```

**Step 7: Access the Application**
```
Open browser and go to: http://localhost:5000
```

---

### Method 2: Quick Install (Without Virtual Environment)

⚠️ Not recommended for production, but works for quick testing

```bash
cd stock_prediction_project
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
python app.py
```

---

### Method 3: Conda Installation

If you prefer Anaconda/Miniconda:

```bash
# Create conda environment
conda create -n stock_forecast python=3.10

# Activate environment
conda activate stock_forecast

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Run application
python app.py
```

---

## 🔧 TROUBLESHOOTING INSTALLATION

### Issue 1: pip install fails

**Error:** `ERROR: Could not find a version that satisfies the requirement`

**Solution:**
```bash
# Update pip
pip install --upgrade pip

# Try installing again
pip install -r requirements.txt

# If still fails, install packages individually
pip install flask pandas numpy scikit-learn tensorflow nltk textblob yfinance
```

### Issue 2: TensorFlow installation fails

**Error:** `Could not install TensorFlow`

**Solution for Windows:**
```bash
# Install Visual C++ Redistributable first
# Download from Microsoft website

# Then install TensorFlow
pip install tensorflow==2.15.0
```

**Solution for macOS (M1/M2):**
```bash
# Use TensorFlow for macOS
pip install tensorflow-macos==2.15.0
pip install tensorflow-metal  # For GPU acceleration
```

**Solution for Linux:**
```bash
# Ensure you have required system libraries
sudo apt-get update
sudo apt-get install python3-dev

# Then install TensorFlow
pip install tensorflow==2.15.0
```

### Issue 3: NLTK download fails

**Error:** `SSL: CERTIFICATE_VERIFY_FAILED`

**Solution:**
```python
# Run this in Python interpreter
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt')
nltk.download('stopwords')
```

### Issue 4: Port 5000 already in use

**Error:** `Address already in use`

**Solution 1 - Change Port:**
Edit `app.py`, change last line:
```python
app.run(debug=True, host='0.0.0.0', port=5001)  # Use different port
```

**Solution 2 - Kill Process:**
```bash
# On Windows:
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# On macOS/Linux:
lsof -ti:5000 | xargs kill -9
```

### Issue 5: Module not found errors

**Error:** `ModuleNotFoundError: No module named 'xyz'`

**Solution:**
```bash
# Make sure virtual environment is activated
# Reinstall all dependencies
pip install -r requirements.txt --force-reinstall
```

---

## 📋 POST-INSTALLATION VERIFICATION

### Run System Tests

```bash
# Navigate to project directory
cd stock_prediction_project

# Run test script
python test_system.py
```

**Expected Output:**
```
======================================================================
STOCK PRICE FORECASTING SYSTEM - COMPONENT TEST
MSc IT Project - University of Mumbai
======================================================================

[TEST 1] Checking Python Version...
✅ Python 3.10 - OK

[TEST 2] Testing Core Library Imports...
✅ Flask
✅ Pandas
✅ NumPy
✅ Scikit-learn
✅ TensorFlow
✅ NLTK
✅ TextBlob
✅ yfinance
✅ Plotly

[TEST 3] Checking Project Structure...
✅ data/raw/
✅ data/processed/
✅ models/
...

[All tests should pass ✅]
```

---

## 🎯 FIRST RUN GUIDE

### Initial Setup Checklist

- [ ] Virtual environment activated
- [ ] All dependencies installed
- [ ] NLTK data downloaded
- [ ] Test script passed
- [ ] Application running
- [ ] Browser opened to localhost:5000

### First Use Workflow

**1. Start Application**
```bash
# Ensure you're in the project directory
cd stock_prediction_project

# Activate virtual environment (if using)
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate      # Windows

# Start Flask
python app.py
```

**2. Open Browser**
```
Navigate to: http://localhost:5000
```

**3. Select Stock**
```
Click on any stock symbol (e.g., AAPL)
```

**4. Fetch Data**
```
Click "📥 Fetch Latest Data"
Wait ~30 seconds
```

**5. Train Models**
```
Click "🎓 Train Models"
Wait 2-5 minutes
```

**6. Make Prediction**
```
Click "🔮 Make Prediction"
View results!
```

---

## 🔒 OPTIONAL: News API Setup

For real news data (optional, system works without it):

**Step 1: Get API Key**
1. Visit: https://newsapi.org/
2. Sign up for free account
3. Get your API key

**Step 2: Configure**
Edit `config.py`:
```python
NEWS_API_KEY = 'your_actual_api_key_here'
```

**Step 3: Test**
Fetch data again and you'll see real news sentiment!

---

## 💾 DATABASE SETUP

Database is created automatically on first run.

**Location:** `database/db.sqlite`

**Manual Reset:**
```bash
# If you need to reset database
rm database/db.sqlite
python app.py  # Will recreate automatically
```

**Backup Database:**
```bash
# Create backup
cp database/db.sqlite database/db.sqlite.backup

# Restore backup
cp database/db.sqlite.backup database/db.sqlite
```

---

## 🖥️ SYSTEM REQUIREMENTS

### Minimum Requirements
- **OS:** Windows 10, macOS 10.14+, Linux (Ubuntu 18.04+)
- **CPU:** Dual-core 2GHz
- **RAM:** 4GB
- **Storage:** 2GB free space
- **Python:** 3.10+
- **Internet:** Required for data fetching

### Recommended Requirements
- **OS:** Windows 11, macOS 12+, Linux (Ubuntu 20.04+)
- **CPU:** Quad-core 2.5GHz+
- **RAM:** 8GB+
- **Storage:** 5GB free space
- **Python:** 3.11
- **Internet:** Broadband connection

---

## 📊 PERFORMANCE OPTIMIZATION

### For Faster Training

**1. Use GPU (if available)**
TensorFlow will automatically use GPU if CUDA is installed.

**2. Reduce Data**
Edit `config.py`:
```python
START_DATE = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')  # 1 year instead of 2
```

**3. Reduce LSTM Epochs**
Edit `config.py`:
```python
LSTM_PARAMS = {
    'epochs': 25,  # Reduced from 50
    ...
}
```

**4. Use Fewer Trees in Random Forest**
Edit `config.py`:
```python
RANDOM_FOREST_PARAMS = {
    'n_estimators': 50,  # Reduced from 100
    ...
}
```

---

## 🌐 NETWORK CONFIGURATION

### Firewall Settings

If application is not accessible:

**Windows:**
```
Control Panel → Windows Defender Firewall → Allow an app
Add Python to allowed apps
```

**macOS:**
```
System Preferences → Security & Privacy → Firewall → Firewall Options
Add Python to allowed apps
```

**Linux:**
```bash
sudo ufw allow 5000/tcp
```

### Access from Other Devices

To access from other computers on same network:

**1. Find your IP:**
```bash
# Windows
ipconfig

# macOS/Linux
ifconfig
```

**2. Run with host='0.0.0.0'**
```python
app.run(debug=True, host='0.0.0.0', port=5000)
```

**3. Access from other device:**
```
http://YOUR_IP_ADDRESS:5000
```

---

## 📱 BROWSER COMPATIBILITY

**Tested and Working:**
- ✅ Chrome 90+
- ✅ Firefox 88+
- ✅ Safari 14+
- ✅ Edge 90+

**Not Recommended:**
- ❌ Internet Explorer (any version)

---

## 🔄 UPDATING THE SYSTEM

### Update Dependencies

```bash
# Activate virtual environment first
pip install --upgrade -r requirements.txt
```

### Update Specific Package

```bash
pip install --upgrade package_name
```

### Check for Outdated Packages

```bash
pip list --outdated
```

---

## 🗑️ UNINSTALLATION

### Remove Project

```bash
# Deactivate virtual environment
deactivate

# Delete project folder
rm -rf stock_prediction_project  # macOS/Linux
rmdir /s stock_prediction_project  # Windows
```

### Remove Virtual Environment Only

```bash
# From within project directory
rm -rf venv  # macOS/Linux
rmdir /s venv  # Windows
```

---

## 🆘 GETTING HELP

### Self-Help Resources

1. **Check README.md** - Comprehensive documentation
2. **Check QUICKSTART.md** - Quick setup guide
3. **Run test_system.py** - Identify issues
4. **Check error messages** - Terminal output
5. **Review VIVA_PREP.md** - Technical details

### Common Issues Database

| Issue | Solution |
|-------|----------|
| Module not found | `pip install -r requirements.txt` |
| Port in use | Change port in app.py |
| NLTK error | Run NLTK download commands |
| Database error | Delete db.sqlite, restart app |
| TensorFlow error | Install Visual C++ (Windows) |

---

## 📚 ADDITIONAL RESOURCES

### Official Documentation

- Python: https://docs.python.org/3/
- Flask: https://flask.palletsprojects.com/
- TensorFlow: https://www.tensorflow.org/
- Scikit-learn: https://scikit-learn.org/
- Pandas: https://pandas.pydata.org/

### Learning Resources

- Machine Learning: https://www.coursera.org/learn/machine-learning
- Flask Tutorial: https://flask.palletsprojects.com/tutorial/
- Stock Analysis: https://www.investopedia.com/

---

## ✅ FINAL CHECKLIST

Before presenting/submitting:

- [ ] System runs without errors
- [ ] All tests pass (test_system.py)
- [ ] Can fetch data successfully
- [ ] Can train models successfully
- [ ] Can make predictions successfully
- [ ] Visualizations display correctly
- [ ] Documentation is complete
- [ ] Code is well-commented
- [ ] Database is working
- [ ] Screenshots taken (for backup)

---

## 🎓 FOR ACADEMIC SUBMISSION

### Required Deliverables

1. **Source Code** ✅
   - All Python files
   - Templates and static files
   - Configuration files

2. **Documentation** ✅
   - README.md
   - QUICKSTART.md
   - VIVA_PREP.md
   - Code comments

3. **Requirements** ✅
   - requirements.txt
   - System requirements documented

4. **Testing** ✅
   - test_system.py
   - Test results

5. **Deployment Guide** ✅
   - This file!

---

## 🎉 CONGRATULATIONS!

You now have a fully functional stock price forecasting system!

**Next Steps:**
1. ✅ Complete installation
2. ✅ Test the system
3. ✅ Prepare for viva (see VIVA_PREP.md)
4. ✅ Practice demonstration
5. ✅ Submit project

**Good luck with your MSc IT project! 🚀📈**

---

**Last Updated:** January 2024  
**Project:** Stock Price Forecasting System  
**Institution:** University of Mumbai  
**Degree:** MSc IT

---
