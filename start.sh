set -o errexit

cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000 &


cd ../frontend
streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
