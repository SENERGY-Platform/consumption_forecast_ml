# Consumption Forecast ML Operator

Der Operator verarbeitet fortlaufende Zählerstände aus Kafka und prognostiziert
den Verbrauch für die aktuelle Stunde, vier Stunden, den aktuellen Tag, die
Woche, den Monat oder das Jahr. Die Kafka-Anbindung, historische Daten und die
Modellverwaltung über MLflow werden von `operator-lib` bereitgestellt.

## Konfiguration

Die operatorspezifische Konfiguration enthält:

- `time_period`: `H`, `4H`, `D`, `W`, `M` oder `Y`
- `model_type`: `prophet` oder `nhits`

Weitere Einstellungen wie `ray_url`, `mlflow_url`, `ts_conn`, Kafka-Topics und
Mappings stammen aus `operator-lib`. Der gemappte Eingangswert muss als
`value` vorliegen; der Zeitstempel wird von der Library separat übergeben.

Die Ausgabe enthält je nach `time_period` beispielsweise
`HourPrediction` und `HourPredictionTotal`. Der Ausgabezeitstempel ist das Ende
des prognostizierten Zeitraums.

## Training und Betrieb

Das Training wird regelmäßig über eine Ray-Remote-Funktion im konfigurierten
Cluster ausgeführt. `operator-lib` verbindet sich dazu mit `ray_url`, richtet
aber kein eigenes Ray-`runtime_env` ein. Das Cluster-Image muss deshalb diesen
Operator und seine Python-Abhängigkeiten bereits enthalten.

Lokaler Start mit Python 3.10.20 und `uv`:

```shell
uv sync
uv run python main.py
```

