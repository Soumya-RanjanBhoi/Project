set -o errexit

# python backend/main.py &

streamlit run frontent/new.py --server.port=$PORT --server.address=0.0.0.0
