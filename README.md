# llm-extractor-kb
LLM based extractor working in collaboration with an evolving Knowledge Base

## 📦 Installation
On the **remote** server, set up the Conda environment:
```bash
# Create environment
conda env create -f environment.yml
conda activate extraction

# (Optional) Download spaCy English model
python -m spacy download en_core_web_lg
```

## 🚀 Usage
### Terminal 1: Start the Streamlit app
On the **remote** server, run:

```bash
conda activate extraction
python -m streamlit run app/main.py
```
or with cached KBs (will load any KBs that have already been scanned and cached, and scan the rest as needed):
```bash
bash run_app.sh
```

If successful, you will see a message like:
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://10.x.x.x:8501
  External URL: http://y.y.y.y:8501
```

### Terminal 2: SSH Port Forwarding
After starting the app, open a new terminal window.

On your **local** machine, forward the Streamlit port (default: 8501).
If the app is running on a different port, use that port number instead.
```
ssh -L 8501:localhost:<streamlit_port> <server_user>@<server_name>
```
Example:
```bash
ssh -L 8501:localhost:8501 name@remote-instance
```

### Access the App
Once the SSH connection is established and the port forwarding is active, open your browser and go to [http://localhost:8501](http://localhost:8501).
