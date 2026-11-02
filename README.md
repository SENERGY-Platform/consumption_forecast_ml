# Consumption Forecast ML Operator

Der Operator verarbeitet fortlaufende Zählerstände und prognostiziert den
Verbrauch der laufenden Stunde, des laufenden 4-Stunden-Blocks, Tages, der
Woche, des Monats oder Jahres. Er löst `SENERGY-Platform/consumption_forecast`
ab und hat dessen Schnittstelle. Die Kafka-Anbindung, historische Daten und die
Modellverwaltung über MLflow werden von `operator-lib` bereitgestellt.

## Eingang

| Name          | Typ   | Beschreibung |
|---------------|-------|--------------|
| `Consumption` | float | Zählerstand  |

Der Zeitpunkt einer Messung ist der Zeitstempel ihrer Kafka-Nachricht.

## Konfiguration

| Name                  | Typ    | Beschreibung                                                   | Standard  |
|-----------------------|--------|----------------------------------------------------------------|-----------|
| `time_periods`        | string | Kommagetrennt: `H`, `4H`, `D`, `W`, `M` und/oder `Y`           | `D`       |
| `model`               | string | `prophet` oder `nhits`                                         | `prophet` |
| `add_time_covariates` | bool   | Kalendermerkmale als Kovariaten: Monat, bei Rastern unter einer Woche der Wochentag, bei stündlichem Raster die Stunde | `false` |
| `logger_level`        | string | Log-Level, etwa `info` oder `debug`                            | `warning` |

Weitere Einstellungen wie `ray_url`, `mlflow_url` und `ts_conn` stammen aus
`operator-lib`; `ts_conn` setzt die Flow-Engine.

## Ausgabe

Je konfigurierter Periode drei Felder, alle Perioden in einer Nachricht.
`<Periode>` ist `Hour`, `FourHour`, `Day`, `Week`, `Month` oder `Year`.

| Name                       | Typ    | Beschreibung |
|----------------------------|--------|--------------|
| `<Periode>Prediction`      | float  | Verbrauch der laufenden Periode: bisher gemessen plus Prognose für den Rest |
| `<Periode>PredictionTotal` | float  | Erwarteter Zählerstand am Periodenende |
| `<Periode>Timestamp`       | string | Ende der Periode in UTC, z. B. `2026-01-14T22:59:59Z` |

Eine Periode wird neu prognostiziert, sobald ihr Raster ein Intervall
weiterrückt oder sie neu beginnt: die Stunde und der 4-Stunden-Block stündlich,
der Tag alle 4 Stunden, Woche und Monat täglich, das Jahr wöchentlich. Die
Nachricht enthält dann auch die letzten Prognosen der übrigen Perioden. Ihr
Zeitstempel ist der der auslösenden Messung.

Die Zeiträume werden nach deutscher Zeit (`Europe/Berlin`) abgegrenzt: Ein Tag
beginnt um 00:00 Uhr Ortszeit, eine Woche am Montag, 4H-Blöcke um 00, 04, 08 …
Uhr. An den Tagen der Zeitumstellung dauert ein Tag 23 bzw. 25 Stunden.

Die Modelle prognostizieren den Verbrauch in einem festen UTC-Raster: stündlich
für `H` und `4H`, in 4-Stunden-Blöcken für `D`, täglich für `W` und `M`,
wöchentlich für `Y`. Ragt ein Rasterintervall über eine Periodengrenze, wird es
zeitanteilig aufgeteilt. Das betrifft nur `D`, `W`, `M` und `Y` und auch dort nur
das erste und letzte Intervall einer Periode.

Die Schnittstelle ist in `op.py` deklariert (`ConsumptionInputs`,
`ForecastOutputs`, `CustomConfig`). Daraus erzeugt
`uv run python -m operator_lib manifest op.py` die Datei
`operator-manifest.json` mit Eingängen, Ausgaben und Konfiguration für den
Eintrag in der Plattform.

## Training und Betrieb

Je Raster wird ein Modell trainiert; zusammen bilden sie das MLflow-Modell des
Operators. Ein Raster wird neu trainiert, wenn sein Modell fehlt, nicht zur
Konfiguration passt oder älter ist als sein Intervall: stündliches und
4-Stunden-Raster täglich, tägliches Raster wöchentlich, wöchentliches alle 28
Tage. Schlägt ein Training fehl, versucht der Operator es frühestens nach einer
Stunde erneut.

Das Training wird über eine Ray-Remote-Funktion im konfigurierten Cluster
ausgeführt. Der Cluster läuft mit den Standard-Images von Ray und enthält weder
diesen Operator noch seine Abhängigkeiten. Läuft der Operator unter `uv run`,
überträgt Ray das Arbeitsverzeichnis samt `pyproject.toml` und `uv.lock` an den
Cluster und baut dort dieselbe Umgebung auf. Das Training läuft damit immer mit
dem Code des Operators; ohne `uv run` schlägt es fehl. Das Docker-Image startet
den Operator deshalb mit `uv run`.

## Image

Jeder Push auf `master` baut
`ghcr.io/senergy-platform/consumption-forecast-operator:vX.Y.Z`. Die Version
ergibt sich aus den Conventional Commits seit dem letzten Versions-Tag. Das
Repo löst `SENERGY-Platform/consumption_forecast` ab, dessen Image-Reihe bei
v1.0.1 endete, und beginnt deshalb mit v2.0.0. `:prod` zeigt bis zur Umstellung
des Operator-Eintrags in der Plattform noch auf den alten Operator.

## Lokale Entwicklung

Mit Python 3.10.20 und `uv`:

```shell
uv sync
uv run python main.py
```

Tests:

```shell
uv run python -m unittest discover -s tests
```
