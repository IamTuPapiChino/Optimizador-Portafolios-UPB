========================================================================
HERRAMIENTA DE ANÁLISIS Y OPTIMIZACIÓN FINANCIERA | Junio 2026
========================================================================

PROYECTO ACADÉMICO: Finanzas I
INSTITUCIÓN: Universidad Privada Boliviana (UPB) - Campus La Paz
AUTOR: Gabriel Fabian Soliz De Ugarte

========================================================================
1. DESCRIPCIÓN GENERAL
========================================================================
La siguiente herramienta consiste en un entorno virtual avanzado desarrollado 
en Python y renderizado mediante Streamlit. La herramienta permite la 
descarga de precios de cierre ya sea vía Yahoo Finance o por la lectura de bases 
de datos locales previamente limpiadas como de Refinitiv. Esta herramienta ejecuta 
análisis descriptivos de riesgo y rentabilidad, modelación de portafolios 
(Frontera Eficiente de Markowitz) y valoración de activos mediante el modelo Capital
Asset Pricing Model (CAPM).

========================================================================
2. ESTRUCTURA DEL DIRECTORIO
========================================================================
El repositorio se encuentra estructurado de la siguiente manera:

* /.venv            : Entorno virtual de Python aislado.
* /codigo/app.py    : Archivo principal que contiene la arquitectura del sistema.
* /datos            : Repositorio para almacenar archivos Excel/CSV de Refinitiv.
* requirements.txt  : Manifiesto de dependencias y librerías necesarias.
* README.txt        : Documento de documentación técnica y despliegue.

========================================================================
3. INSTRUCCIONES DE DESPLIEGUE, CREACIÓN Y EJECUCIÓN
========================================================================
Para ejecutar la aplicación en un entorno local desde cero, se debe 
cumplir con la siguiente secuencia de comandos en la terminal de Visual 
Studio Code o la consola del sistema:

Paso A: Creación del Entorno Virtual
Para garantizar el aislamiento de las librerías, se debe generar un entorno 
virtual nuevo ejecutando el siguiente comando:
> python -m venv .venv

Paso B: Activación del Entorno
Una vez creada la carpeta oculta .venv, se procede a su activación:
> .venv\Scripts\activate

[NOTA TÉCNICA: RESOLUCIÓN DE BLOQUEOS EN POWERSHELL]
En sistemas operativos Windows, es frecuente que las políticas de seguridad 
estrictas de PowerShell bloqueen la activación de scripts arrojando un error 
indicando que la ejecución está deshabilitada ("running scripts 
is disabled on this system"). 

Para solucionar este bloqueo, se debe ejecutar el siguiente comando para 
otorgar permisos de ejecución exclusivamente al usuario actual:
> Set-ExecutionPolicy Unrestricted -Scope CurrentUser

Al presionar "Enter", el sistema puede solicitar confirmación, para lo cual 
se debe presionar la tecla "S" (Sí) o "Y" (Yes). Una vez aprobada la política, 
se debe intentar ejecutar nuevamente el comando del Paso B.

Paso C: Instalación de Dependencias
Con el entorno virtual activado de forma exitosa (marcado por el prefijo 
"(.venv)" en la terminal), se procede a instalar el listado de paquetes:
> pip install -r requirements.txt

Paso D: Lanzamiento de la Aplicación
Finalizada la descarga de dependencias, se debe inicializar el servidor 
local de Streamlit apuntando al archivo principal del código fuente:
> streamlit run codigo/app.py

========================================================================
4. INSTRUCCIONES DE CARGA DE DATOS OPERATIVOS
========================================================================
* Fuente Yahoo Finance: Seleccionar la opción en la barra lateral e ingresar 
  los tickers bursátiles correspondientes separados por comas.
* Fuente Local (Refinitiv): Guardar el documento Excel de origen dentro de 
  la carpeta /datos de este directorio. Seleccionar la opción de carga local 
  en la interfaz y cargar el archivo.

========================================================================