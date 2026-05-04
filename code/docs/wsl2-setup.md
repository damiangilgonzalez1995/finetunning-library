# WSL2 + Ubuntu para entornos de Deep Learning en Windows

Guía práctica para tener un entorno Linux dentro de Windows, listo para entrenar modelos con CUDA y PyTorch. Esto es lo que necesitas si quieres usar herramientas como **Unsloth, axolotl, vLLM o llama.cpp** que oficialmente solo soportan Linux.

---

## ¿Qué es WSL2?

**Windows Subsystem for Linux 2** es una capa de Microsoft que permite correr una distribución Linux REAL (kernel Linux completo, no emulado) **dentro de Windows**, sin dual-boot ni máquinas virtuales pesadas.

| | WSL1 | WSL2 | VirtualBox / VMware | Dual boot |
|---|---|---|---|---|
| Kernel | traducido | **real** | real | real |
| Velocidad I/O | lento | rápido | medio | nativo |
| GPU passthrough | no | **sí (NVIDIA)** | difícil | nativo |
| Setup | fácil | fácil | medio | complejo |
| Coexistencia con Windows | sí | sí | sí | NO |

Para Deep Learning con NVIDIA, **WSL2 es la opción correcta** desde 2021. Permite usar tu GPU desde Linux mientras sigues con Windows abierto.

---

## Pre-requisitos

- **Windows 10** versión 21H2 o superior, **o Windows 11** (cualquier versión).
- **GPU NVIDIA** (cualquiera con CUDA compute capability 5.2+).
- Driver NVIDIA en Windows con soporte WSL2 (todos los recientes lo tienen, desde 2021).
- Permisos de administrador la primera vez (para activar la feature).

Para comprobar tu versión de Windows: `Win+R` → `winver` → mira "Versión".

---

## Paso 1: Activar e instalar WSL2 (un solo comando)

Abre **PowerShell como administrador** (Win+X → Terminal (Admin) en Win11, o Win+X → PowerShell (Admin) en Win10).

```powershell
wsl --install -d Ubuntu-22.04
```

Este único comando:
1. Activa la feature de Windows "Subsystem for Linux".
2. Activa la feature "Virtual Machine Platform" (necesaria para WSL2).
3. Descarga el kernel de WSL2.
4. Descarga e instala Ubuntu 22.04 desde Microsoft Store.
5. Configura WSL2 como default.

**Tarda 5-10 min**. Cuando termine puede pedirte reiniciar Windows. Reinicia.

> Si falla con "WSL ya está habilitado", lo más probable es que solo te quede instalar la distro Ubuntu por separado. Ver paso 2 alternativo abajo.

### Alternativa: si WSL ya está habilitado pero no tienes Ubuntu

Si `wsl --list --verbose` te muestra distros existentes (por ejemplo `docker-desktop`) pero no Ubuntu real:

```powershell
wsl --install -d Ubuntu-22.04
```

Solo descarga Ubuntu (las features ya están). O alternativamente, abre **Microsoft Store**, busca "Ubuntu 22.04 LTS" e instala desde la UI.

---

## Paso 2: Primer arranque de Ubuntu

Tras la instalación, **se abre automáticamente una ventana negra** con Ubuntu pidiéndote:

1. **Username** (ej. `damian`, en minúsculas, sin espacios). Este NO tiene que coincidir con tu usuario de Windows.
2. **Password** (apúntalo, lo necesitarás para `sudo`). Cuando lo escribes **no ves los caracteres** (es normal en terminales Linux).

Cuando termines verás un prompt verde tipo:
```
damian@DESKTOP-XXX:~$
```

Ese `damian@DESKTOP-XXX` es tu usuario y el host. El `~` es tu home (`/home/damian`).

---

## Paso 3: Verificar que la GPU se ve desde WSL2

Esto es CRÍTICO. Sin esto, no puedes usar Unsloth/PyTorch/cualquier framework GPU.

Dentro de Ubuntu:

```bash
nvidia-smi
```

Deberías ver tu GPU (modelo, VRAM, driver, CUDA version). Si la ves → **passthrough OK, todo bien**.

### Si NO la ves

Tres posibles causas:

| Síntoma | Causa | Solución |
|---|---|---|
| `nvidia-smi: command not found` | Las utilidades NVIDIA no están instaladas | `sudo apt install -y nvidia-utils-535` |
| `Failed to initialize NVML: Driver/library version mismatch` | Driver de Windows desactualizado | Actualiza desde https://www.nvidia.com/Download/index.aspx |
| `NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver` | Lo mismo | Igual, actualizar driver Windows + reiniciar |

---

## Paso 4: Setup básico de Python para Deep Learning

Ubuntu 22.04 viene con Python 3.10. Para DL es preferible Python 3.11 o 3.12.

```bash
# Instalar utilidades de Python + git + build tools
sudo apt update
sudo apt install -y python3-pip python3-venv python3-dev git build-essential

# Verificar
python3 --version
```

### Crear un venv para tus proyectos de DL

**Buena práctica**: un venv por proyecto (no instalar nada globalmente).

```bash
cd ~
python3 -m venv mi-proyecto-env
source mi-proyecto-env/bin/activate     # ACTIVAR el venv

# A partir de aquí tu prompt cambia a (mi-proyecto-env):
# Y todo lo que instales con pip va a este venv aislado
pip install --upgrade pip
pip install torch transformers unsloth   # o lo que necesites
```

**Importante**: el `source ... activate` solo dura mientras la terminal esté abierta. Al cerrar y volver a abrir Ubuntu, hay que repetir el comando.

### Auto-activar un venv al abrir terminal

Si solo trabajas con un venv principal, añádelo al final de `~/.bashrc`:

```bash
echo 'source ~/mi-proyecto-env/bin/activate' >> ~/.bashrc
```

A partir de la próxima sesión, el venv se activa solo.

---

## Paso 5: Acceder a archivos entre Windows y Ubuntu

Los dos filesystems son visibles entre sí:

| Desde | Cómo accede al otro |
|---|---|
| Ubuntu (WSL2) | `/mnt/c/Users/TuUsuario/...` (todo el disco C: aparece como `/mnt/c/`) |
| Windows | `\\wsl$\Ubuntu-22.04\home\tuusuario\...` (en explorador de archivos) |

### Decisión importante: ¿dónde poner los datos del proyecto?

| Ubicación | Velocidad I/O | Cuándo usar |
|---|---|---|
| `/mnt/c/...` (disco Windows) | **lento** | si los archivos también los necesitas desde Windows |
| `~/...` (filesystem nativo Ubuntu) | **rápido** | datasets de entrenamiento, archivos solo usados desde Linux |

**Para entrenamiento**: si vas a leer miles de imágenes durante un training, hospedar el dataset en `~/` (filesystem Ubuntu nativo) puede ser **2-3x más rápido** que en `/mnt/c/...`. Para proyectos pequeños la diferencia no se nota.

---

## Paso 6: Abrir VSCode desde WSL2

VSCode tiene una extensión llamada **WSL** (Microsoft) que conecta el editor Windows al filesystem Ubuntu, ejecutando el kernel del notebook dentro de Linux.

### Setup

1. En VSCode (Windows), instala la extensión **"WSL"** (publisher: Microsoft).
2. Reinicia VSCode.

### Uso

Desde la terminal de Ubuntu:

```bash
cd /ruta/a/tu/proyecto
code .
```

Esto abre VSCode en modo WSL, conectado a Ubuntu, con esa carpeta como workspace. La primera vez tarda ~30 segundos (instala el VSCode Server en Linux).

Confirmas que está bien conectado por la etiqueta verde abajo a la izquierda: **"WSL: Ubuntu-22.04"**.

### IMPORTANTE: las extensiones se instalan POR contexto

Las extensiones que tienes en VSCode-Windows **NO se aplican** cuando estás en VSCode-WSL. Tienes que instalarlas otra vez en el contexto WSL:

1. Panel de extensiones (`Ctrl+Shift+X`)
2. Busca "Python" (Microsoft) → botón **"Install in WSL: Ubuntu"**
3. Lo mismo para "Jupyter" si vas a usar notebooks

---

## Comandos útiles del día a día

```bash
# Listar distros instaladas y su estado
wsl --list --verbose

# Apagar todas las distros (libera RAM)
wsl --shutdown

# Reiniciar una distro
wsl --terminate Ubuntu-22.04

# Cambiar la distro por defecto (si tienes varias)
wsl --set-default Ubuntu-22.04
```

---

## Troubleshooting común

### "WSL2 está consumiendo mucha RAM"

WSL2 reserva RAM agresivamente. Para limitar el máximo, crea `C:\Users\TuUsuario\.wslconfig`:

```
[wsl2]
memory=8GB        # ajusta a la mitad de tu RAM total
processors=4
swap=2GB
```

Reinicia WSL: `wsl --shutdown` y vuelve a abrir Ubuntu.

### "El terminal de Ubuntu no se abre"

Reinicia la distro: `wsl --shutdown` en PowerShell, luego abre Ubuntu otra vez.

### "VSCode-WSL no encuentra mi venv"

Selecciona el intérprete manualmente: `Ctrl+Shift+P` → `Python: Select Interpreter` → `Enter interpreter path...` → pega la ruta completa: `/home/damian/mi-venv/bin/python`.

### "Quiero borrar Ubuntu y empezar de cero"

```powershell
wsl --unregister Ubuntu-22.04
wsl --install -d Ubuntu-22.04
```

(Se pierde TODO lo que tenías en Ubuntu — datasets, configuración, todo.)

---

## Recursos

- [Documentación oficial Microsoft WSL](https://learn.microsoft.com/en-us/windows/wsl/)
- [Guía NVIDIA CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
- [VSCode + WSL extension](https://code.visualstudio.com/docs/remote/wsl)
