#!/bin/bash

DATA_URL="https://www.kaggle.com/api/v1/datasets/download/arnabbiswas1/microsoft-azure-predictive-maintenance"

echo "-- Utilidad de descarga de datos desde Kaggle --"

echo "Info: Descargando datos ..."

curl -L -o ./data.zip\
  $DATA_URL

if ! ls ./data.zip 1> /dev/null 2>&1; then
    echo "Error: Ocurrió un error durante la descarga"
else
    echo "Info: Descomprimiendo ..."
    unzip -o ./data.zip -d ./raw
    rm ./data.zip
fi

if ! ls ./raw/*.csv 1> /dev/null 2>&1; then
    echo "Error: No se encontraron archivos CSV en la carpeta raw."
else
    echo "Info: Descarga completada"
fi