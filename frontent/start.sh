set -o errexit

streamlit run app.py --server.port=$PORT --server.address=0.0.0.0

