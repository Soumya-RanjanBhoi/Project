set -o errexit

python backend/main.py &

streamlit run frontend/app.py --server.port=$PORT --server.address=0.0.0.0
