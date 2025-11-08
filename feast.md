# Health check
curl http://localhost:6566/health

# Status check
curl http://localhost:6566/status

# Store data
curl -X POST http://localhost:6566/store-data \
  -H "Content-Type: application/json" \
  -d '{"data": [{"TransactionID": 1, "TransactionAmt": 100.0, "timestamp": "2025-11-08T00:00:00"}]}'


  docker exec -it fraud-detection-airflow-webserver cat /opt/airflow/simple_auth_manager_passwords.json.generated


# Move DAG to correct location
mv preprocessing_pipeline.py dags/
docker logs airflow-scheduler --tail 50
docker exec airflow-scheduler ls -la /opt/airflow/dags/
docker exec airflow-scheduler airflow dags list
docker exec airflow-scheduler airflow dags list-import-errors
docker exec airflow-scheduler python -c "import sys; sys.path.insert(0, '/opt/airflow/dags'); from preprocessing_pipeline import dag; print('DAG imported successfully:', dag.dag_id)"
docker restart airflow-scheduler airflow-webserver
docker restart airflow-scheduler airflow-webserver
docker exec airflow-scheduler airflow dags list
docker exec airflow-scheduler airflow dags reserialize
