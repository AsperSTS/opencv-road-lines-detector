# Procesamiento de Video para Resaltar Líneas de Carretera

Este proyecto utiliza OpenCV para procesar videos y resaltar líneas de carretera mediante el uso de técnicas de procesamiento de imágenes. Permite visualizar el video en tiempo real o exportarlo como un archivo de video.

## Requisitos

- Python 3.x
- OpenCV
- Numpy
- argparse

Puedes instalar las dependencias necesarias usando pip:

```bash
pip install opencv-python numpy argparse
```

## Ejecución del Programa

Para ejecutar el programa, usa la siguiente línea de comandos:

```bash
python main.py -m <modo> -v <ruta_del_video>
```

## Parametros

**-m o --modo: Este parámetro define el modo de ejecución.**

1. Ver video en tiempo real.
2. Exportar video como archivo MP4 (modo por defecto).

**-v o --video: La ruta del video que deseas procesar. Este parámetro es obligatorio.**
