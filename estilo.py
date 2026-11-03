# Estilo común de todas las figuras del informe: paleta azul-violeta (monocromática) y tipografía de paper.
# Cada notebook que dibuja hace `from estilo import *` y listo, así cambiar un color acá cambia todas las figuras.
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# --- Paleta: de azul marino a violeta claro ---
AZUL_OSCURO = "#0F2A6B"     # XGBoost (el modelo de referencia)
AZUL_CLARO = "#3E86D9"      # GP
VIOLETA = "#7A2FA6"         # SNGP
VIOLETA_CLARO = "#C08ADF"   # Ensamble

COLORES = {"XGBoost": AZUL_OSCURO, "GP": AZUL_CLARO, "GP ajustado": AZUL_CLARO,
           "SNGP": VIOLETA, "Ensamble": VIOLETA_CLARO}
ESTILOS = {"XGBoost": "-", "GP": "-", "GP ajustado": ":", "SNGP": "-", "Ensamble": "--"}   # además del color, cambia la línea
RELLENO = {"XGBoost": True, "GP": True, "GP ajustado": True, "SNGP": True, "Ensamble": False}   # el ensamble va con marcador hueco
MARCAS = {"XGBoost": "o", "GP": "s", "GP ajustado": "s", "SNGP": "^", "Ensamble": "D"}      # y el marcador (se lee en blanco y negro)

COLOR_RESTO = "#1F3F99"     # segmento "resto" (azul)
COLOR_THIN = "#8E3AB5"      # segmento "thin-file" (violeta)
BUENO = "#6F9BDB"           # buen pagador (azul medio)
MOROSO = "#8E1F8F"          # moroso (violeta magenta, bien distinto del azul)

# Mapas de color: probabilidad (azul - blanco - violeta) e incertidumbre (blanco a violeta oscuro)
CMAP_CLASES = LinearSegmentedColormap.from_list("clases", ["#2F4B9A", "#FFFFFF", "#A24BB0"])
CMAP_INCERT = plt.get_cmap("magma_r")   # incertidumbre: de amarillo pálido a negro pasando por violeta


def aplicar_estilo():
    """Tipografía serif con símbolos estilo LaTeX, ejes finos, grilla suave y fuentes editables en el PDF."""
    plt.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "cm", "font.size": 8,
        "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "xtick.direction": "in", "ytick.direction": "in", "axes.linewidth": 0.6,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def guardar(fig, nombre):
    """Guarda la figura en pdf (para el informe) y png (para redes), en outputs/."""
    fig.savefig(f"outputs/{nombre}.pdf", bbox_inches="tight")
    fig.savefig(f"outputs/{nombre}.png", dpi=300, bbox_inches="tight")


aplicar_estilo()
