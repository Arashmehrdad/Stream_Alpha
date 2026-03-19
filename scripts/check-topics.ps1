docker exec -it streamalpha-redpanda rpk topic list -X brokers=redpanda:9092
docker exec -it streamalpha-redpanda rpk topic consume raw.trades -n 5 --offset start -X brokers=redpanda:9092
docker exec -it streamalpha-redpanda rpk topic consume raw.ohlc -n 5 --offset start -X brokers=redpanda:9092
docker exec -it streamalpha-redpanda rpk topic consume raw.health -n 5 --offset start -X brokers=redpanda:9092
