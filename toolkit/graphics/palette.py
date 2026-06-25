import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style()

### set rcParams
plt.rcParams["text.usetex"] = True
plt.rcParams["figure.dpi"] = 100
plt.rcParams["figure.figsize"] = (10,6)

# font sizes
plt.rcParams["font.size"] = 12
plt.rcParams["axes.titlesize"] = 20
plt.rcParams["axes.labelsize"] = 26
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 12
# plt.rcParams['axes.titlesize'] = 'large'

SECTOR_COLORS = {
    "Irrigation": "#4c7d0d",
    "Municipal": "#006490",
    "Mining": "#888888",
    "Industry": "#7d0d7c",
    "Power": "#664e00", 
}

SECTOR_NAMES = {
    "IRR": "Irrigation",
    "MUN": "Municipal",
    "MIN": "Mining",
    "IND": "Industry",
    "POW": "Power", 
}

DROUGHT_PALETTE = sns.color_palette("rocket")
DROUGHT_PALETTE.reverse()

MEANPROPS={"marker": "o",
            "markeredgecolor": "black",
            "markerfacecolor": "white",
            "markersize": "10"}

rcparams = {'legend.fontsize': 'x-large',
          'figure.figsize': (10, 6),
         'axes.labelsize': 20,
         'axes.titlesize': 20,
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}

plt.rcParams.update(rcparams)
