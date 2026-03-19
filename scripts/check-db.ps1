docker exec -it streamalpha-postgres psql -U streamalpha -d streamalpha -c "SELECT COUNT(*) AS trade_rows FROM raw_trades;"
docker exec -it streamalpha-postgres psql -U streamalpha -d streamalpha -c "SELECT COUNT(*) AS ohlc_rows FROM raw_ohlc;"
docker exec -it streamalpha-postgres psql -U streamalpha -d streamalpha -c "SELECT service_name, status, last_event_at FROM producer_heartbeat;"
