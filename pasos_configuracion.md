# Pasos para configurar folder de darknet

## Archivos necesarios
- Carpeta data: archivos .obj de configuración
- Carpeta training: pesos del entrenamiento
- Makefile: modificaciones para Jetson
- Archivo .cfg: configuración de la red
- Archivo .py: Corre la detección en un video
- Video: Para utilizarlo en la prueba de detección

## 1. Agregar nvcc al path (~/.bashrc)

    export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64\
                    ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

## 2. Descargar el repositorio de darknet
    
    git clone https://github.com/AlexeyAB/darknet.git

## 3. Copiar el archivo Makefile al folder de darknet

    cp Makefile ./darknet/

## 4. Ejecutar el comando make
Utiliza los siguientes comandos

    cd darknet
    make

Ignora los warnings, si no se muestra un error, todo salió bien.

## 5. Copiar la carpeta data y el archivo .cfg al folder de darknet
Utiliza los siguientes comandos

    rm -r ./data/
    cp -r ../data/ .

## 6. Correr el script deteccion_video.py

    cd ..
    python3 deteccion_video.py