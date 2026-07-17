pipeline {
  agent any

  options {
    disableConcurrentBuilds()
    timeout(time: 4, unit: 'HOURS')
    timestamps()
  }

  parameters {
    string(name: 'AIRFLOW_URL', defaultValue: 'http://airflow:8080', description: 'Internal Airflow URL.')
    string(name: 'AIRFLOW_DAG_ID', defaultValue: 'cashlog33_training_pipeline', description: 'Promotion-gated CashLog DAG.')
  }

  stages {
    stage('Trigger Airflow') {
      steps {
        withCredentials([usernamePassword(credentialsId: 'airflow-local-basic', usernameVariable: 'AIRFLOW_USER', passwordVariable: 'AIRFLOW_PASSWORD')]) {
          sh '''
            set -eu
            DAG_RUN_ID="jenkins-${JOB_NAME}-${BUILD_NUMBER}"
            DAG_RUN_ID=$(printf '%s' "$DAG_RUN_ID" | tr '/ :' '---')
            printf '%s' "$DAG_RUN_ID" > .airflow-run-id
            curl --fail --silent --show-error \
              --user "$AIRFLOW_USER:$AIRFLOW_PASSWORD" \
              -H 'Content-Type: application/json' \
              -X POST \
              "$AIRFLOW_URL/api/v1/dags/$AIRFLOW_DAG_ID/dagRuns" \
              -d "{\"dag_run_id\":\"$DAG_RUN_ID\",\"conf\":{\"source\":\"jenkins\",\"build_url\":\"$BUILD_URL\"}}" \
              > airflow-trigger.json
            jq -e '.dag_run_id' airflow-trigger.json
          '''
        }
      }
    }

    stage('Monitor Training') {
      steps {
        withCredentials([usernamePassword(credentialsId: 'airflow-local-basic', usernameVariable: 'AIRFLOW_USER', passwordVariable: 'AIRFLOW_PASSWORD')]) {
          sh '''
            set -eu
            DAG_RUN_ID=$(cat .airflow-run-id)
            while true; do
              curl --fail --silent --show-error \
                --user "$AIRFLOW_USER:$AIRFLOW_PASSWORD" \
                "$AIRFLOW_URL/api/v1/dags/$AIRFLOW_DAG_ID/dagRuns/$DAG_RUN_ID" \
                > airflow-status.json
              STATE=$(jq -r '.state' airflow-status.json)
              printf 'Airflow DAG %s state=%s\n' "$DAG_RUN_ID" "$STATE"
              case "$STATE" in
                success) break ;;
                failed) exit 1 ;;
              esac
              sleep 20
            done
          '''
        }
      }
    }

    stage('Archive Decision') {
      steps {
        sh '''
          set -eu
          mkdir -p archived-reports
          cp airflow-trigger.json airflow-status.json archived-reports/
          if [ -d /catai-reports/cashlog33/airflow_latest ]; then
            cp -R /catai-reports/cashlog33/airflow_latest archived-reports/
          fi
        '''
        archiveArtifacts artifacts: 'archived-reports/**', fingerprint: true
      }
    }
  }
}
