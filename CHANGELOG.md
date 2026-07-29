# Changelog

## 0.2.1

- Poprawka nazwy encji binary sensora pojazdu: `binary_sensor.ha_plate_recognizer_vehicle` (była błędna `vehicle_detected`)

## 0.2.0

- Dodano binary sensor `binary_sensor.ha_plate_recognizer_vehicle` (ON/OFF) via MQTT discovery
- Sensor wykrywa obecność pojazdu w kadrze i umożliwia automatyzacje HA (np. powiadomienie gdy auto stoi min. 5 sekund)
- Stan publikowany tylko przy zmianie (retained), nie generuje zbędnych wiadomości MQTT

## 0.1.9

- Filtr kierunku ruchu (`direction_filter`, `entry_direction`) — ignoruje auta wyjeżdżające
- Parametr `motion_min_px` do filtrowania minimalnego ruchu w kadrze

## 0.1.8

- Integracja z `input_select.plates` — lista znanych tablic pobierana dynamicznie z HA
- Snapshot zapisywany do `/share/plate_recognizer` przy dopasowaniu tablicy
- Parametr `plates_select_entity` w konfiguracji

## 0.1.7

- Pierwsze publiczne wydanie
- Detekcja pojazdów przez MediaPipe
- Odczyt tablic rejestracyjnych przez fast-alpr
- Publikacja wyników przez MQTT
- MQTT Discovery dla Home Assistant
- Web UI z podglądem na żywo (port konfigurowalny)
- ROI (Region of Interest) konfigurowalny przez opcje
- Cooldown między dopasowaniami tablic
