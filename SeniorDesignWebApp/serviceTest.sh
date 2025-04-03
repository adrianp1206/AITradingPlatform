#!/usr/bin/env bash
clear
echo "Purging SQS queue"
aws sqs purge-queue --queue-url "https://sqs.us-east-2.amazonaws.com/183631322250/GabrielBrown-Standard1" --region "us-east-2"

sleep 5

echo "Starting Finnhub Service in a new process"
gradle finhub-service:runServerWithMockFinhubPreload &
finnhubServicePid=$!
echo "Finnhub Service PID is $finnhubServicePid"

sleep 5

echo "Starting Monolith Service in a new process"
gradle monolith-service:runService &
monolithServicePid=$!
# capture output of integration test
testResult=$?
echo "Monolith Service PID is $monolithServicePid"
sleep 15

echo "Waiting for services to come up"

echo "Starting server integration tests"
gradle executeServerIntegrationTest

sleep 2

echo "Stopping all services"
kill -9 $finnhubServicePid
kill -9 $monolithServicePid
# exit with result of integration test and not the above kill command which would have been the default
exit $testResult

